# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime I/O plumbing for the native planner.

This module contains **zero decision logic**.  It only gathers data from the
outside world (Prometheus, FPM subscribers, K8s connectors) and applies
scaling decisions back.  All scaling logic lives in
:class:`~dynamo.planner.core.state_machine.PlannerScalingState` and is exposed
through builtin orchestrator plugins.

Subclasses (PrefillPlanner, DecodePlanner, AggPlanner, DisaggPlanner) set
mode-specific flags and override ``_bootstrap_regression`` and
``_apply_effects``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import aiohttp.web
from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException
from prometheus_client import start_http_server

from dynamo.planner.config.backend_components import get_planner_k8s_component_names
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY, _initialize_gpu_counts
from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    PlannerEffects,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError, PlannerError
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.live_dashboard import start_live_dashboard
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.planner.monitoring.traffic_metrics import Metrics, PrometheusAPIClient
from dynamo.planner.monitoring.worker_info import WorkerInfo, resolve_worker_info
from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
    from dynamo.llm import FpmEventSubscriber

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

ConnectorType = Union[GlobalPlannerConnector, KubernetesConnector, VirtualConnector]

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers for building WorkerCapabilities from resolved WorkerInfo
# ------------------------------------------------------------------


def _engine_caps(
    worker_info: Optional[WorkerInfo], num_gpu: Optional[int]
) -> Optional[EngineCapabilities]:
    if worker_info is None and num_gpu is None:
        return None
    return EngineCapabilities(
        num_gpu=num_gpu,
        max_num_batched_tokens=(
            worker_info.max_num_batched_tokens if worker_info else None
        ),
        max_num_seqs=worker_info.max_num_seqs if worker_info else None,
        context_length=worker_info.context_length if worker_info else None,
        max_kv_tokens=worker_info.max_kv_tokens if worker_info else None,
        kv_cache_block_size=worker_info.kv_cache_block_size if worker_info else None,
        speculative_nextn=worker_info.speculative_nextn if worker_info else None,
    )


def build_worker_capabilities(
    config: PlannerConfig,
    prefill_worker_info: Optional[WorkerInfo] = None,
    decode_worker_info: Optional[WorkerInfo] = None,
) -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=_engine_caps(prefill_worker_info, config.prefill_engine_num_gpu),
        decode=_engine_caps(decode_worker_info, config.decode_engine_num_gpu),
    )


# ------------------------------------------------------------------
# Base adapter
# ------------------------------------------------------------------


class NativePlannerBase:
    """Base adapter: runtime I/O plumbing shared by all planner modes.

    Subclasses set ``require_prefill`` / ``require_decode`` and override
    ``_bootstrap_regression()`` and ``_apply_effects()``.
    """

    require_prefill: bool = False
    require_decode: bool = False

    def __init__(
        self, runtime: Optional[DistributedRuntime], config: PlannerConfig
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.namespace = config.namespace
        self.runtime_namespace = config.namespace
        self.model_name: Optional[str] = None

        # Connector
        self.connector: ConnectorType
        if config.environment == "global-planner":
            assert config.global_planner_namespace is not None
            assert runtime is not None
            self.connector = GlobalPlannerConnector(
                runtime,
                self.namespace,
                config.global_planner_namespace,
                "GlobalPlanner",
                config.model_name,
            )
        elif config.environment == "kubernetes":
            self.connector = KubernetesConnector(self.namespace, config.model_name)
        elif config.environment == "virtual":
            assert runtime is not None
            self.connector = VirtualConnector(
                runtime, self.namespace, config.model_name
            )
        else:
            raise ValueError(f"Invalid environment: {config.environment}")

        # Prometheus
        self.prometheus_traffic_client = PrometheusAPIClient(
            config.metric_pulling_prometheus_endpoint,
            config.namespace,
            metrics_source=config.throughput_metrics_source,
            bearer_token=config.metric_pulling_prometheus_token,
            bearer_token_file=config.metric_pulling_prometheus_token_file,
            ssl_verify=config.metric_pulling_prometheus_ssl_verify,
            extra_query_params=config.metric_pulling_prometheus_extra_query_params,
            ca_bundle=config.metric_pulling_prometheus_ca_bundle,
        )
        if config.throughput_metrics_source == "router":
            self.prometheus_traffic_client.warn_if_router_not_scraped()

        self.prometheus_port = config.metric_reporting_prometheus_port
        self.prometheus_metrics = PlannerPrometheusMetrics()
        if self.prometheus_port != 0:
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    f"Started Prometheus metrics server on port {self.prometheus_port}"
                )
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")
            self.prometheus_metrics.sla_target_ttft_ms.set(config.ttft_ms)
            self.prometheus_metrics.sla_target_itl_ms.set(config.itl_ms)

        # Worker info (resolved during _async_init)
        self.prefill_worker_info = WorkerInfo()
        self.decode_worker_info = WorkerInfo()

        # FPM subscribers (one per component type, populated during _async_init)
        self._prefill_fpm_sub: Optional[FpmEventSubscriber] = None
        self._decode_fpm_sub: Optional[FpmEventSubscriber] = None

        # Runtime client caches
        self._prefill_client = None
        self._decode_client = None

        # Shared metrics state
        self._last_metrics = Metrics()
        self._cumulative_gpu_hours: float = 0.0
        self._last_gpu_hours_update_ts: Optional[float] = None

        # Power-annotation sweep throttle (Phase 1). The sweep reconciles pod
        # power-limit annotations against K8s; running it every tick is needless
        # apiserver load, so run() throttles it to
        # config.power_annotation_interval_seconds and forces a per-tick sweep
        # for one interval after each scale-up (when new pods appear).
        self._last_power_annotation_sweep_s: float = 0.0
        self._force_power_annotations_until_s: float = 0.0

        # Diagnostics recorder
        self._recorder = DiagnosticsRecorder(config=config)

        # Live dashboard runner (started in _async_init)
        self._dashboard_runner: Optional[aiohttp.web.AppRunner] = None

        # Tick engine: the main-loop dispatch target. It wraps the builtin
        # plugin pipeline via ``OrchestratorEngineAdapter``.
        self._engine: Optional[EngineProtocol] = None

        # Cached worker counts from the most recent tick's input — lets
        # ``_log_decision_summary`` read current replica counts without
        # reaching into engine internals.
        self._last_worker_counts: Optional[WorkerCounts] = None

    # ------------------------------------------------------------------
    # Engine bootstrap
    # ------------------------------------------------------------------

    async def _install_benchmark_fpms(
        self,
        *,
        prefill_fpms=None,
        decode_fpms=None,
        agg_fpms=None,
    ) -> None:
        """Install benchmark FPMs into the builtin orchestrator engine.

        Mode subclasses call this from ``_bootstrap_regression`` with
        whatever FPM subset their mode produces. The builtin orchestrator
        loads those FPMs into its shared scaling state and then runs the
        plugin Bootstrap pass.

        Skipping ``None`` kwargs preserves mode-specific semantics:
        PrefillPlanner passes only ``prefill_fpms``; DisaggPlanner may
        pass one or both depending on ``fetch_pre_deployment_metrics``
        outcomes; AggPlanner passes ``agg_fpms``.
        """
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        engine = self._ensure_engine()
        assert isinstance(engine, OrchestratorEngineAdapter)
        await engine.bootstrap_from_fpms(
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
            agg_fpms=agg_fpms,
            historical_traffic=self._load_predictor_warmup_observations(),
        )

    def _ensure_engine(self) -> EngineProtocol:
        """Lazy-construct the builtin-orchestrator tick engine."""
        if self._engine is not None:
            return self._engine
        caps = build_worker_capabilities(
            self.config,
            self.prefill_worker_info,
            self.decode_worker_info,
        )
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        self._engine = OrchestratorEngineAdapter(self.config, caps)
        return self._engine

    def _load_predictor_warmup_observations(
        self,
    ) -> Optional[list[TrafficObservation]]:
        if self.config.load_predictor_warmup_trace is None:
            return None
        try:
            metrics = extract_metrics_from_mooncake(
                self.config.load_predictor_warmup_trace,
                self.config.throughput_adjustment_interval_seconds,
            )
            return [
                TrafficObservation(
                    duration_s=self.config.throughput_adjustment_interval_seconds,
                    num_req=float(m["request_count"]),
                    isl=float(m["avg_isl"]),
                    osl=float(m["avg_osl"]),
                )
                for m in metrics
            ]
        except Exception as e:
            logger.warning(f"Failed to warm load predictors: {e}")
            return None

    async def _bootstrap_engine_plugins_if_needed(self) -> None:
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        engine = self._ensure_engine()
        if isinstance(engine, OrchestratorEngineAdapter):
            if engine.plugins_bootstrapped:
                return
            await engine.bootstrap_plugins(
                historical_traffic=self._load_predictor_warmup_observations()
            )

    # ------------------------------------------------------------------
    # Async init
    # ------------------------------------------------------------------

    async def _async_init(self) -> None:
        if hasattr(self, "connector") and hasattr(self.connector, "_async_init"):
            await self.connector._async_init()

        # Resolve DGD component names once for this backend+mode and store
        # them so _refresh_worker_info_from_connector can reuse them without
        # recomputing.  In agg mode the decode slot resolves to the agg worker
        # name (e.g. VllmWorker) rather than the disagg decode name.
        self._prefill_k8s_name, self._decode_k8s_name = get_planner_k8s_component_names(
            self.config.backend, self.config.mode
        )

        logger.info("Validating deployment...")
        await self.connector.validate_deployment(
            prefill_component_name=(
                self._prefill_k8s_name if self.require_prefill else None
            ),
            decode_component_name=(
                self._decode_k8s_name if self.require_decode else None
            ),
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
        )
        logger.info("Successfully validated the deployment")
        _initialize_gpu_counts(
            self.config,
            self.connector,
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
            prefill_component_name=self._prefill_k8s_name
            if self.require_prefill
            else None,
            decode_component_name=self._decode_k8s_name
            if self.require_decode
            else None,
        )
        await self.connector.wait_for_deployment_ready(include_planner=False)
        await self._refresh_runtime_namespace()

        # Resolve WorkerInfo once from the connector.  For K8s this populates
        # runtime_config fields from MDC CRDs; for Virtual it returns backend
        # defaults (subscribers aren't attached yet) which is enough to
        # construct the FPM endpoint.
        await self._init_worker_info()

        if self.runtime is not None:
            if self.require_prefill:
                await self._init_fpm_subscriber("prefill")
            if self.require_decode:
                await self._init_fpm_subscriber("decode")

        # VirtualConnector reads MDC from the FPM subscriber's discovery watch;
        # hand it the subscribers now that they exist.  The tick-loop refresh
        # will backfill runtime_config fields once discovery sees the workers.
        if isinstance(self.connector, VirtualConnector):
            self.connector.set_mdc_subscribers(
                prefill=self._prefill_fpm_sub,
                decode=self._decode_fpm_sub,
            )

        await self._bootstrap_regression()
        await self._bootstrap_engine_plugins_if_needed()

        # Log operating mode at startup
        if self.config.advisory:
            logger.info(
                "[ADVISORY] Planner started in advisory mode — "
                "scaling decisions will be logged but NOT executed."
            )

        # Start live dashboard if configured
        if self.config.live_dashboard_port:
            try:
                self._dashboard_runner = await start_live_dashboard(
                    self._recorder, self.config.live_dashboard_port
                )
            except Exception as e:
                logger.error(f"Failed to start live dashboard: {e}")

    async def _init_worker_info(self) -> None:
        connector = getattr(self, "connector", None)
        self.prefill_worker_info, self.decode_worker_info = resolve_worker_info(
            backend=self.config.backend,
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
            connector=connector,
            config_model_name=getattr(self.config, "model_name", ""),
            no_operation=False,
            prefill_component_name=getattr(self, "_prefill_k8s_name", None),
            decode_component_name=getattr(self, "_decode_k8s_name", None),
        )
        self.model_name = (
            self.decode_worker_info.model_name or self.prefill_worker_info.model_name
        )

    def _resolve_runtime_namespace(self) -> str:
        if hasattr(self.connector, "get_worker_runtime_namespace"):
            return self.connector.get_worker_runtime_namespace(  # type: ignore[attr-defined]
                self.namespace
            )
        return self.namespace

    async def _refresh_runtime_namespace(self) -> None:
        """Refresh worker runtime namespace and rebind runtime handles if needed."""
        try:
            runtime_namespace = self._resolve_runtime_namespace()
        except (ApiException, ConfigException, PlannerError) as e:
            logger.warning(
                f"Failed to resolve worker runtime namespace from connector: {e}; "
                f"keeping current runtime namespace {self.runtime_namespace}"
            )
            return
        if runtime_namespace == self.runtime_namespace:
            return

        old_namespace = self.runtime_namespace
        self.runtime_namespace = runtime_namespace
        self._prefill_client = None
        self._decode_client = None
        if self._prefill_fpm_sub is not None:
            self._prefill_fpm_sub.shutdown()
        if self._decode_fpm_sub is not None:
            self._decode_fpm_sub.shutdown()
        self._prefill_fpm_sub = None
        self._decode_fpm_sub = None
        logger.info(
            f"Worker runtime namespace changed: {old_namespace} -> {runtime_namespace}"
        )

        if self.runtime is None or self.model_name is None:
            return
        if self.require_prefill:
            await self._init_fpm_subscriber("prefill")
        if self.require_decode:
            await self._init_fpm_subscriber("decode")

    async def _init_fpm_subscriber(self, component: str) -> None:
        from dynamo.llm import FpmEventSubscriber

        worker_info = (
            self.prefill_worker_info
            if component == "prefill"
            else self.decode_worker_info
        )
        if not worker_info.component_name or not worker_info.endpoint:
            logger.warning(
                f"WorkerInfo missing for {component}, cannot create FPM subscriber"
            )
            return

        assert self.runtime is not None
        endpoint = self.runtime.endpoint(
            f"{self.runtime_namespace}.{worker_info.component_name}.{worker_info.endpoint}"
        )
        sub = FpmEventSubscriber(endpoint)
        sub.start_tracking()
        logger.info(
            f"FPM tracker started for "
            f"{self.runtime_namespace}.{worker_info.component_name}.{worker_info.endpoint}"
        )

        if component == "prefill":
            self._prefill_fpm_sub = sub
        else:
            self._decode_fpm_sub = sub

    async def _bootstrap_regression(self) -> None:
        """Override in subclasses to bootstrap regression models."""
        pass

    # ------------------------------------------------------------------
    # Discovery refresh
    # ------------------------------------------------------------------

    _MDC_REFRESH_FIELDS = (
        "total_kv_blocks",
        "kv_cache_block_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "context_length",
        "speculative_nextn",
    )

    def _refresh_worker_info_from_connector(self) -> None:
        """Re-query the connector for any sub-component whose WorkerInfo is
        still missing runtime-config fields.

        This handles the cold-start path where workers haven't registered
        their model cards yet when ``_init_worker_info`` first runs.  It is
        a no-op for K8s mode once CRDs are present, and drives the
        VirtualConnector's discovery-sourced population once cards arrive.
        """
        if not hasattr(self.connector, "get_worker_info"):
            return

        targets: list[tuple[WorkerInfo, SubComponentType]] = []
        if self.require_prefill:
            targets.append((self.prefill_worker_info, SubComponentType.PREFILL))
        if self.require_decode:
            targets.append((self.decode_worker_info, SubComponentType.DECODE))

        changed = False
        for worker_info, sub_type in targets:
            if worker_info.max_num_batched_tokens is not None:
                continue
            component_name = (
                getattr(self, "_prefill_k8s_name", None)
                if sub_type == SubComponentType.PREFILL
                else getattr(self, "_decode_k8s_name", None)
            )
            try:
                if component_name is None:
                    fresh = self.connector.get_worker_info(
                        sub_type, self.config.backend
                    )
                else:
                    fresh = self.connector.get_worker_info(
                        sub_type, self.config.backend, component_name=component_name
                    )
            except Exception as e:
                logger.debug(
                    f"get_worker_info refresh for {sub_type.value} failed: {e}"
                )
                continue

            updated = False
            for field_name in self._MDC_REFRESH_FIELDS:
                fresh_val = getattr(fresh, field_name)
                if (
                    fresh_val is not None
                    and getattr(worker_info, field_name) != fresh_val
                ):
                    setattr(worker_info, field_name, fresh_val)
                    updated = True
            if updated:
                changed = True
                logger.info(
                    f"Refreshed {sub_type.value} WorkerInfo from connector: "
                    f"{worker_info.summary()}"
                )

        if changed:
            capabilities = build_worker_capabilities(
                self.config,
                self.prefill_worker_info,
                self.decode_worker_info,
            )
            update_engine_capabilities = getattr(
                self._engine, "update_capabilities", None
            )
            if callable(update_engine_capabilities):
                update_engine_capabilities(capabilities)

    # ------------------------------------------------------------------
    # Data collection (runtime I/O)
    # ------------------------------------------------------------------

    def _decode_fpm_bytes(
        self, subscriber: Optional[FpmEventSubscriber]
    ) -> dict[tuple[str, int], ForwardPassMetrics]:
        from dynamo.common.forward_pass_metrics import decode as decode_fpm

        if subscriber is None:
            return {}
        result = {}
        for key, raw_bytes in subscriber.get_recent_stats().items():
            fpm = decode_fpm(raw_bytes)
            if fpm is not None:
                result[key] = fpm
        return result

    async def _get_or_create_client(self, component_name: str, endpoint_name: str):
        assert self.runtime is not None
        client = await self.runtime.endpoint(
            f"{self.runtime_namespace}.{component_name}.{endpoint_name}"
        ).client()
        await asyncio.sleep(0.1)
        return client

    async def _get_worker_counts_raw(self) -> tuple[int, int, bool]:
        """Returns (num_prefill, num_decode, is_stable) from connector or runtime."""
        if hasattr(self, "connector") and hasattr(
            self.connector, "get_actual_worker_counts"
        ):
            (
                prefill_count,
                decode_count,
                is_stable,
            ) = self.connector.get_actual_worker_counts(
                prefill_component_name=(
                    self.prefill_worker_info.k8s_name if self.require_prefill else None
                ),
                decode_component_name=(
                    self.decode_worker_info.k8s_name if self.require_decode else None
                ),
            )
            return (
                prefill_count if self.require_prefill else 0,
                decode_count if self.require_decode else 0,
                is_stable,
            )

        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")

        num_p, num_d = 0, 0
        if self.require_prefill:
            try:
                if self._prefill_client is None:
                    assert self.prefill_worker_info.component_name is not None
                    assert self.prefill_worker_info.endpoint is not None
                    self._prefill_client = await self._get_or_create_client(
                        self.prefill_worker_info.component_name,
                        self.prefill_worker_info.endpoint,
                    )
                num_p = len(self._prefill_client.instance_ids())  # type: ignore
            except Exception:
                logger.warning("No prefill workers found")

        if self.require_decode:
            try:
                if self._decode_client is None:
                    assert self.decode_worker_info.component_name is not None
                    assert self.decode_worker_info.endpoint is not None
                    self._decode_client = await self._get_or_create_client(
                        self.decode_worker_info.component_name,
                        self.decode_worker_info.endpoint,
                    )
                num_d = len(self._decode_client.instance_ids())  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to get decode worker endpoints: {e}")

        return num_p, num_d, True

    async def _collect_traffic(self) -> Optional[TrafficObservation]:
        """Pull traffic metrics from Prometheus over the throughput interval."""
        assert self.model_name is not None
        interval_str = f"{self.config.throughput_adjustment_interval_seconds}s"
        m = self._last_metrics
        m.ttft = (
            self.prometheus_traffic_client.get_avg_time_to_first_token(
                interval_str, self.model_name
            )
            * 1000
        )
        m.itl = (
            self.prometheus_traffic_client.get_avg_inter_token_latency(
                interval_str, self.model_name
            )
            * 1000
        )
        m.num_req = self.prometheus_traffic_client.get_avg_request_count(
            interval_str, self.model_name
        )
        m.request_duration = self.prometheus_traffic_client.get_avg_request_duration(
            interval_str, self.model_name
        )
        m.isl = self.prometheus_traffic_client.get_avg_input_sequence_tokens(
            interval_str, self.model_name
        )
        m.osl = self.prometheus_traffic_client.get_avg_output_sequence_tokens(
            interval_str, self.model_name
        )
        m.kv_hit_rate = self.prometheus_traffic_client.get_avg_kv_hit_rate(
            interval_str, self.model_name
        )
        m.accept_length = self._collect_accept_length(interval_str)

        normalized_idle_metrics = m.normalize_idle_nans()
        if normalized_idle_metrics:
            logger.info(
                "Zero traffic observed; treating undefined averages as 0: %s",
                ", ".join(normalized_idle_metrics),
            )

        hit_rate_str = f"{m.kv_hit_rate:.3f}" if m.kv_hit_rate is not None else "n/a"
        accept_length_str = (
            f"{m.accept_length:.3f}" if m.accept_length is not None else "n/a"
        )
        logger.info(
            f"Observed num_req: {m.num_req:.2f} isl: {m.isl:.2f} osl: {m.osl:.2f} "
            f"kv_hit_rate: {hit_rate_str} accept_length: {accept_length_str}"
        )

        if self.prometheus_port != 0:
            self.prometheus_metrics.observed_ttft_ms.set(m.ttft)
            self.prometheus_metrics.observed_itl_ms.set(m.itl)
            self.prometheus_metrics.observed_requests_per_second.set(
                m.num_req / self.config.throughput_adjustment_interval_seconds
            )
            self.prometheus_metrics.observed_request_duration_seconds.set(
                m.request_duration
            )
            self.prometheus_metrics.observed_input_sequence_tokens.set(m.isl)
            self.prometheus_metrics.observed_output_sequence_tokens.set(m.osl)

        if not m.is_valid():
            logger.info("Metrics contain None or NaN values, skipping")
            return None
        return TrafficObservation(
            duration_s=self.config.throughput_adjustment_interval_seconds,
            num_req=m.num_req,
            isl=m.isl,
            osl=m.osl,
            kv_hit_rate=m.kv_hit_rate,
            accept_length=m.accept_length,
        )

    def _collect_accept_length(self, interval_str: str) -> Optional[float]:
        if self.config.mode not in ("disagg", "decode", "agg"):
            return None
        if not self.model_name:
            return None
        component_name = self.decode_worker_info.component_name
        endpoint_name = self.decode_worker_info.endpoint
        return self.prometheus_traffic_client.get_avg_spec_decode_accept_length(
            interval_str,
            self.config.backend,
            component_name,
            self.model_name,
            namespace=self.runtime_namespace,
            endpoint_name=endpoint_name,
        )

    async def _collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        """Pull only the KV hit rate from Prometheus over ``duration_s``.

        Used in load-only deployments: the load tick only needs the hit rate
        to discount prefill work, so we skip the five other (unused) traffic
        queries to keep the per-load-tick scrape cheap.

        The observation is still returned when metrics are unavailable so the
        planner can retain last-value runtime metadata without skipping
        the load tick.
        """
        assert self.model_name is not None
        if duration_s <= 0:
            return None
        interval_str = f"{int(duration_s)}s"
        hit_rate = self.prometheus_traffic_client.get_avg_kv_hit_rate(
            interval_str, self.model_name
        )
        accept_length = self._collect_accept_length(interval_str)
        # Mirror the observed value into Metrics so the diagnostics recorder
        # sees the up-to-date hit rate even on load-only ticks.
        self._last_metrics.kv_hit_rate = hit_rate
        self._last_metrics.accept_length = accept_length
        hit_rate_str = f"{hit_rate:.3f}" if hit_rate is not None else "n/a"
        accept_length_str = (
            f"{accept_length:.3f}" if accept_length is not None else "n/a"
        )
        logger.info(
            f"Observed kv_hit_rate over {interval_str}: {hit_rate_str}; "
            f"accept_length: {accept_length_str}"
        )
        return TrafficObservation(
            duration_s=duration_s,
            num_req=0.0,
            isl=0.0,
            osl=0.0,
            kv_hit_rate=hit_rate,
            accept_length=accept_length,
        )

    def _collect_fpm(self) -> FpmObservations:
        """Collect FPM from active subscribers."""
        prefill_stats = None
        decode_stats = None

        if self._prefill_fpm_sub is not None:
            stats = self._decode_fpm_bytes(self._prefill_fpm_sub)
            if stats:
                for (wid, dp), fpm in stats.items():
                    _log_fpm(wid, dp, fpm, "prefill")
                prefill_stats = stats

        if self._decode_fpm_sub is not None:
            stats = self._decode_fpm_bytes(self._decode_fpm_sub)
            if stats:
                for (wid, dp), fpm in stats.items():
                    _log_fpm(wid, dp, fpm, "decode")
                decode_stats = stats

        if self.prometheus_port != 0:
            self._emit_per_engine_fpm(prefill_stats, decode_stats)

        return FpmObservations(prefill=prefill_stats, decode=decode_stats)

    def _emit_per_engine_fpm(
        self,
        prefill_stats: Optional[dict] = None,
        decode_stats: Optional[dict] = None,
    ) -> None:
        pm = self.prometheus_metrics
        pm.engine_queued_prefill_tokens.clear()
        pm.engine_queued_decode_kv_tokens.clear()
        pm.engine_inflight_decode_kv_tokens.clear()

        if prefill_stats:
            for (wid, dp), fpm in prefill_stats.items():
                labels = dict(worker_id=wid, dp_rank=str(dp))
                pm.engine_queued_prefill_tokens.labels(**labels).set(
                    fpm.queued_requests.sum_prefill_tokens
                )

        if decode_stats:
            for (wid, dp), fpm in decode_stats.items():
                labels = dict(worker_id=wid, dp_rank=str(dp))
                pm.engine_queued_decode_kv_tokens.labels(**labels).set(
                    fpm.queued_requests.sum_decode_kv_tokens
                )
                pm.engine_inflight_decode_kv_tokens.labels(**labels).set(
                    fpm.scheduled_requests.sum_decode_kv_tokens
                )

    async def _collect_worker_counts(self) -> WorkerCounts:
        num_p, num_d, is_stable = await self._get_worker_counts_raw()
        expected_p = num_p if is_stable else None
        expected_d = num_d if is_stable else None
        return WorkerCounts(
            ready_num_prefill=num_p if self.require_prefill else None,
            ready_num_decode=num_d if self.require_decode else None,
            expected_num_prefill=expected_p if self.require_prefill else None,
            expected_num_decode=expected_d if self.require_decode else None,
            prefill_scaling_in_progress=self.require_prefill and not is_stable,
            decode_scaling_in_progress=self.require_decode and not is_stable,
        )

    # ------------------------------------------------------------------
    # Gather tick input
    # ------------------------------------------------------------------

    async def _gather_tick_input(self, tick: ScheduledTick) -> TickInput:
        now = time.time()
        traffic = None
        worker_counts = None
        fpm_obs = None

        if tick.need_traffic_metrics:
            # Throughput ticks pull the full traffic snapshot over the
            # throughput interval. Load-only deployments instead piggyback
            # a cheap kv-hit-rate-only scrape (over the load interval) on
            # each load tick so the planner can still discount prefill work
            # by recent prefix reuse.
            if tick.use_full_traffic_metrics:
                traffic = await self._collect_traffic()
            else:
                traffic = await self._collect_kv_hit_rate_observation(
                    tick.traffic_metrics_duration_s
                )
        if tick.need_worker_states:
            worker_counts = await self._collect_worker_counts()
        if tick.need_worker_fpm:
            fpm_obs = self._collect_fpm()

        return TickInput(
            now_s=now,
            traffic=traffic,
            worker_counts=worker_counts,
            fpm_observations=fpm_obs,
        )

    # ------------------------------------------------------------------
    # Apply effects (override in subclasses for mode-specific metrics)
    # ------------------------------------------------------------------

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        """Override in subclasses to report metrics and apply scaling."""
        pass

    async def _apply_power_annotations(self) -> None:
        """Annotate worker pods with per-GPU power limits (Phase 1).

        Reads the actual annotation from each Pod object returned by
        get_component_pods(). Only PATCHes when annotation is missing or wrong.
        K8s is the source of truth — no local cache. The DGD is read once per
        sweep and shared across the prefill/decode pod lookups.

        Performs one full reconcile sweep. The caller
        (_should_sweep_power_annotations in run()) throttles this to
        config.power_annotation_interval_seconds in steady state and forces a
        per-tick sweep for one interval after a scale-up, so newly-created pods
        pick up the cap promptly while steady-state apiserver load stays
        bounded. AIC-driven cap changes (Phase 3+) propagate on the next sweep.

        Skipped in advisory mode — annotations are a cluster-visible side
        effect on customer-owned Pod objects, so advisory runs must observe
        the contract that ``_apply_scaling_targets`` already follows: log
        intent, mutate nothing.

        The kubernetes client is synchronous, so the sweep itself runs in a
        worker thread via ``asyncio.to_thread``. The run loop still waits for
        the reconcile to finish before the next tick, but a slow apiserver or
        a large DGD does not block unrelated asyncio tasks such as metrics,
        diagnostics, and shutdown handling.
        """
        if self.config.advisory:
            return
        connector = self.connector
        if not isinstance(connector, KubernetesConnector):
            return
        if not (self.require_prefill or self.require_decode):
            return
        if not self.config.enable_power_awareness:
            await asyncio.to_thread(self._run_power_annotation_removal, connector)
            return

        await asyncio.to_thread(self._run_power_annotation_sweep, connector)

    def _run_power_annotation_removal(self, connector: KubernetesConnector) -> None:
        """Remove planner-owned power annotations from surviving worker pods.

        Called when power awareness is disabled so that pods previously
        annotated by the planner are cleaned up.  The Power Agent observes the
        missing key and releases the GPU cap through its normal release path.
        """
        try:
            deployment = connector.kube_api.get_graph_deployment(
                connector.graph_deployment_name
            )
            prefill_name, decode_name = get_planner_k8s_component_names(
                self.config.backend, self.config.mode
            )

            pods_to_clean: list = []
            if self.require_prefill:
                pods_to_clean.extend(
                    connector.get_component_pods(
                        SubComponentType.PREFILL,
                        deployment=deployment,
                        component_name=prefill_name,
                    )
                )
            if self.require_decode:
                pods_to_clean.extend(
                    connector.get_component_pods(
                        SubComponentType.DECODE,
                        deployment=deployment,
                        component_name=decode_name,
                    )
                )
        except (ApiException, DynamoGraphDeploymentNotFoundError) as e:
            logger.warning(
                "Power-annotation removal skipped: failed to read DGD or list "
                "worker pods (%s); retrying on the next sweep.",
                e,
            )
            return

        for pod in pods_to_clean:
            if (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY) is None:
                continue
            try:
                connector.kube_api.remove_pod_annotation(
                    pod.metadata.name, POWER_ANNOTATION_KEY
                )
                logger.info(
                    "Removed power annotation from pod %s",
                    pod.metadata.name,
                )
            except ApiException as e:
                if e.status == 409:
                    logger.debug(
                        "409 Conflict removing power annotation from pod %s; "
                        "will retry on next sweep.",
                        pod.metadata.name,
                    )
                elif e.status == 429:
                    logger.warning(
                        "429 TooManyRequests removing power annotation from pod %s; "
                        "no Retry-After backoff, will retry on next sweep.",
                        pod.metadata.name,
                    )
                else:
                    logger.warning(
                        "Failed to remove power annotation from pod %s: %s",
                        pod.metadata.name,
                        e,
                    )

    def _run_power_annotation_sweep(self, connector: KubernetesConnector) -> None:
        """Blocking power-annotation reconcile sweep (runs off the event loop).

        Invoked only via ``asyncio.to_thread`` from
        ``_apply_power_annotations`` after its guards pass. The connector is
        passed explicitly so the worker thread uses the locally narrowed
        ``KubernetesConnector`` rather than relying on a production assert.
        """

        # One DGD read for the whole sweep: resolving prefill and decode pods
        # each needs the deployment, so fetch it once and reuse it rather than
        # letting each get_component_pods() issue its own GET.
        #
        # The DGD read and pod listings hit the apiserver and can fail
        # transiently (ApiException — 5xx/timeout) or when the DGD is
        # momentarily unresolvable (DynamoGraphDeploymentNotFoundError — a
        # 404 blip). This reconciliation sweep is optional and idempotent, so
        # those must not propagate: run() wraps the tick loop in try/finally,
        # so an escaping exception would shut the engine down and exit the
        # planner. Log and skip — the next sweep retries. The catch stays
        # narrow (not the PlannerError base): component/validation errors are
        # already swallowed by get_component_pods(); other unexpected planner
        # subclasses still surface as bugs.
        try:
            deployment = connector.kube_api.get_graph_deployment(
                connector.graph_deployment_name
            )

            # Fallback component names so aggregated (mode=agg) DGDs — whose
            # single worker is labelled ``type: worker`` rather than
            # ``type: prefill/decode`` — still get annotated.  In agg mode the
            # worker is named e.g. ``VllmWorker`` (agg_worker_k8s_name), not
            # ``VllmDecodeWorker``; the fallback is a no-op for disagg (the
            # role matches by type and the explicit name is ignored).
            prefill_name, decode_name = get_planner_k8s_component_names(
                self.config.backend, self.config.mode
            )

            pods_and_limits: list[tuple] = []
            if self.require_prefill:
                for pod in connector.get_component_pods(
                    SubComponentType.PREFILL,
                    deployment=deployment,
                    component_name=prefill_name,
                ):
                    pods_and_limits.append(
                        (pod, str(self.config.prefill_engine_gpu_power_limit))
                    )
            if self.require_decode:
                for pod in connector.get_component_pods(
                    SubComponentType.DECODE,
                    deployment=deployment,
                    component_name=decode_name,
                ):
                    pods_and_limits.append(
                        (pod, str(self.config.decode_engine_gpu_power_limit))
                    )
        except (ApiException, DynamoGraphDeploymentNotFoundError) as e:
            logger.warning(
                "Power-annotation sweep skipped: failed to read DGD or list "
                "worker pods (%s); retrying on the next sweep.",
                e,
            )
            return

        for pod, limit_str in pods_and_limits:
            current = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
            if current == limit_str:
                continue
            try:
                connector.kube_api.patch_pod_annotation(
                    pod.metadata.name, POWER_ANNOTATION_KEY, limit_str
                )
                logger.info(
                    "Annotated pod %s with %s=%s",
                    pod.metadata.name,
                    POWER_ANNOTATION_KEY,
                    limit_str,
                )
            except ApiException as e:
                if e.status == 409:
                    logger.debug(
                        "409 Conflict patching power annotation on pod %s; "
                        "will retry on next sweep.",
                        pod.metadata.name,
                    )
                elif e.status == 429:
                    logger.warning(
                        "429 TooManyRequests patching power annotation on pod %s; "
                        "no Retry-After backoff, will retry on next sweep.",
                        pod.metadata.name,
                    )
                else:
                    logger.warning(
                        "Failed to patch power annotation on pod %s: %s",
                        pod.metadata.name,
                        e,
                    )

    def _current_worker_counts(self) -> tuple[int, int]:
        """Best-known current (prefill, decode) ready worker counts.

        Reads ``self._last_worker_counts``, cached each tick in ``run()`` from
        the tick input — the count source the orchestrator engine exposes.
        Returns ``(0, 0)`` before the first worker-state tick populates it.
        Shared by ``_scaling_up`` and ``_log_decision_summary`` so the two
        can't drift.
        """
        if self._last_worker_counts is not None:
            return (
                self._last_worker_counts.ready_num_prefill or 0,
                self._last_worker_counts.ready_num_decode or 0,
            )
        return 0, 0

    def _scaling_up(self, effects: PlannerEffects) -> bool:
        """True when this tick's decision raises a managed replica count.

        A scale-up means the operator will create new worker pods, which start
        without the power-limit annotation; the caller uses this to force a
        prompt annotation sweep instead of waiting out the steady-state
        throttle. Reads current counts via ``_current_worker_counts``, the
        same source ``_log_decision_summary`` uses, so the force decision
        stays consistent with the logged summary.
        Scale-downs and holds return False because they create no pods.
        """
        if self._last_worker_counts is None:
            return False
        decision = effects.scale_to
        if decision is None:
            return False
        current_p, current_d = self._current_worker_counts()
        if decision.num_prefill is not None and decision.num_prefill > current_p:
            return True
        if decision.num_decode is not None and decision.num_decode > current_d:
            return True
        return False

    def _should_sweep_power_annotations(
        self, now_s: float, effects: PlannerEffects
    ) -> bool:
        """Decide whether to run a power-annotation sweep this tick.

        Gates both the enabled write sweep and the disabled removal sweep.
        Throttles steady-state sweeps to at most one per
        ``power_annotation_interval_seconds`` (a sweep costs a DGD read + a pod
        list per managed component, so per-tick sweeps multiply apiserver
        load). A scale-up (re)opens a force window of one interval during which
        every tick sweeps, so freshly-created pods are annotated without waiting
        out the throttle. When power awareness is disabled the scale-up force
        window is skipped (no new pods are being annotated), but the normal
        throttle still fires so stale annotations are removed on the interval.
        Updates the sweep bookkeeping as a side effect when it returns True.
        """
        if self.config.advisory:
            return False
        if self.config.enable_power_awareness and self._scaling_up(effects):
            self._force_power_annotations_until_s = (
                now_s + self.config.power_annotation_interval_seconds
            )
        within_force_window = now_s < self._force_power_annotations_until_s
        throttle_elapsed = (
            now_s - self._last_power_annotation_sweep_s
            >= self.config.power_annotation_interval_seconds
        )
        if within_force_window or throttle_elapsed:
            self._last_power_annotation_sweep_s = now_s
            return True
        return False

    def _publish_power_budget_metrics(self, num_p: int, num_d: int) -> None:
        """Emit power budget gauges (Phase 1, dashboard observability only).

        These gauges are advisory: they expose the configured budget and the
        statically projected draw for dashboards. They do NOT gate scaling —
        budget enforcement is Phase 2 (PR #9684). Values come from static
        config, not DCGM, so the gauges stay meaningful even when DCGM is down.
        """
        if self.prometheus_port == 0 or not self.config.enable_power_awareness:
            return
        if self.config.total_gpu_power_limit is None:
            return

        p_gpu = self.config.prefill_engine_num_gpu or 0
        d_gpu = self.config.decode_engine_num_gpu or 0
        if (self.require_prefill and p_gpu == 0) or (
            self.require_decode and d_gpu == 0
        ):
            logger.warning(
                "power_projected_watts will be zero or partial: "
                "prefill_engine_num_gpu=%s, decode_engine_num_gpu=%s are not set. "
                "Set these fields to get accurate power budget metrics.",
                self.config.prefill_engine_num_gpu,
                self.config.decode_engine_num_gpu,
            )
        projected = (
            num_p * self.config.prefill_engine_gpu_power_limit * p_gpu
            + num_d * self.config.decode_engine_gpu_power_limit * d_gpu
        )
        budget = self.config.total_gpu_power_limit
        pm = self.prometheus_metrics
        pm.power_budget_total_watts.set(budget)
        pm.power_projected_watts.set(projected)
        pm.power_budget_utilization.set(projected / budget if budget > 0 else 0.0)

    async def _apply_scaling_targets(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        """Shared helper: send scaling targets to connector.

        Skipped in advisory mode (decisions are logged but not executed).
        """
        if self.config.advisory or not targets:
            return
        await self.connector.set_component_replicas(targets, blocking=blocking)

    # ------------------------------------------------------------------
    # Periodic decision summary
    # ------------------------------------------------------------------

    def _log_decision_summary(self, effects: PlannerEffects) -> None:
        """Log a one-line summary of the scaling decision after each tick.

        Current worker counts come from ``_current_worker_counts`` — the
        cached ``self._last_worker_counts`` set in ``run()`` — shared with
        ``_scaling_up`` so the logged summary and the sweep-force decision
        cannot drift.
        """
        decision = effects.scale_to
        diag = effects.diagnostics

        current_p, current_d = self._current_worker_counts()

        rec_p = decision.num_prefill if decision else None
        rec_d = decision.num_decode if decision else None

        delta_p = (rec_p - current_p) if rec_p is not None else 0
        delta_d = (rec_d - current_d) if rec_d is not None else 0

        if decision is None or (delta_p == 0 and delta_d == 0):
            action = "hold"
        elif (delta_p > 0 or delta_d > 0) and (delta_p < 0 or delta_d < 0):
            action = "rebalance"
        elif delta_p > 0 or delta_d > 0:
            action = "scale_up"
        else:
            action = "scale_down"

        logger.info(
            "[summary] %s | current: prefill=%d decode=%d | "
            "recommended: prefill=%s decode=%s (delta: %+d / %+d) | "
            "load_reason=%s throughput_reason=%s | "
            "est_ttft=%.1fms est_itl=%.1fms",
            action.upper(),
            current_p,
            current_d,
            rec_p if rec_p is not None else "-",
            rec_d if rec_d is not None else "-",
            delta_p,
            delta_d,
            diag.load_decision_reason or "n/a",
            diag.throughput_decision_reason or "n/a",
            diag.estimated_ttft_ms or 0,
            diag.estimated_itl_ms or 0,
        )

    # ------------------------------------------------------------------
    # Diagnostics reporting (shared across all adapters)
    # ------------------------------------------------------------------

    def _publish_inventory_and_gpu_hours(self, tick_input: TickInput) -> None:
        """Publish replica counts and cumulative gpu_hours every tick.

        Sourced from tick_input.worker_counts (populated every tick via
        need_worker_states=True); independent of enable_throughput_scaling
        so non-SLA planners also report inventory and cost accounting.
        ``_cumulative_gpu_hours`` is updated regardless of Prometheus
        port so the HTML recorder / live dashboard stay accurate even
        when Prometheus export is disabled.
        """
        if tick_input.worker_counts is None:
            return
        num_p = tick_input.worker_counts.ready_num_prefill or 0
        num_d = tick_input.worker_counts.ready_num_decode or 0

        now = tick_input.now_s
        if self._last_gpu_hours_update_ts is not None:
            dt_s = max(0.0, now - self._last_gpu_hours_update_ts)
            self._cumulative_gpu_hours += (
                (
                    num_p * (self.config.prefill_engine_num_gpu or 0)
                    + num_d * (self.config.decode_engine_num_gpu or 0)
                )
                * dt_s
                / 3600.0
            )
        self._last_gpu_hours_update_ts = now

        if self.prometheus_port == 0:
            return
        self.prometheus_metrics.num_prefill_replicas.set(num_p)
        self.prometheus_metrics.num_decode_replicas.set(num_d)
        self.prometheus_metrics.gpu_hours.set(self._cumulative_gpu_hours)
        self._publish_power_budget_metrics(num_p, num_d)

    @staticmethod
    def _set_if_observed(gauge, value: Optional[float]) -> None:
        """None = no new observation this tick -> leave the gauge unchanged.

        A concrete value (including 0.0) = asserted observation -> publish it.
        Mirrors the set/unset semantics documented on ``PredictionData``.
        """
        if value is not None:
            gauge.set(value)

    def _report_diagnostics(self, tick: ScheduledTick, diag: TickDiagnostics) -> None:
        if self.prometheus_port == 0:
            return
        pm = self.prometheus_metrics
        interval = self.config.throughput_adjustment_interval_seconds

        self._set_if_observed(pm.estimated_ttft_ms, diag.estimated_ttft_ms)
        self._set_if_observed(pm.estimated_itl_ms, diag.estimated_itl_ms)

        if tick.run_load_scaling:
            pm.load_scaling_decision.state(diag.load_decision_reason or "unset")

        # Predicted-load / engine-capacity gauges: ``PredictionData`` supports
        # partial predictions, so each gauge is gated on its own diagnostic --
        # a tick that asserts only some fields must not zero the others. Fields
        # are nulled by _reset_diag() at every tick start, so `is not None`
        # means "produced this tick" (builtin throughput loop or an
        # independently-scheduled PREDICT plugin).
        self._set_if_observed(
            pm.predicted_requests_per_second,
            diag.predicted_num_req / interval
            if diag.predicted_num_req is not None and interval > 0
            else None,
        )
        self._set_if_observed(pm.predicted_input_sequence_tokens, diag.predicted_isl)
        self._set_if_observed(pm.predicted_output_sequence_tokens, diag.predicted_osl)
        self._set_if_observed(
            pm.engine_prefill_capacity_requests_per_second, diag.engine_rps_prefill
        )
        self._set_if_observed(
            pm.engine_decode_capacity_requests_per_second, diag.engine_rps_decode
        )
        # The throughput-decision enum stays gated on the builtin throughput
        # cadence: it reflects a builtin-loop decision, not a plugin prediction.
        if tick.run_throughput_scaling:
            pm.throughput_scaling_decision.state(
                diag.throughput_decision_reason or "unset"
            )

    @staticmethod
    def _should_emit_tick_diagnostics(
        tick: ScheduledTick, effects: PlannerEffects
    ) -> bool:
        """Return True for ticks that should affect operator diagnostics.

        The plugin pipeline may run more frequently than load / throughput
        builtin plugins. No-op pipeline ticks should not overwrite
        Prometheus gauges, add blank report rows, or spam summary logs.
        """
        diag = effects.diagnostics
        return (
            tick.run_load_scaling
            or tick.run_throughput_scaling
            or effects.scale_to is not None
            or bool(diag.audit_events)
            or bool(diag.short_circuit_reason)
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        engine = self._ensure_engine()
        next_tick = engine.initial_tick(time.time())
        poll_interval = self.config.load_adjustment_interval_seconds / 10

        try:
            while True:
                now = time.time()
                if now < next_tick.at_s:
                    await asyncio.sleep(min(next_tick.at_s - now, poll_interval))
                    continue

                await self._refresh_runtime_namespace()
                self._refresh_worker_info_from_connector()

                tick_input = await self._gather_tick_input(next_tick)
                self._publish_inventory_and_gpu_hours(tick_input)
                # Cache worker counts for _log_decision_summary and the
                # power-annotation sweep; None when the tick doesn't request
                # them.
                if tick_input.worker_counts is not None:
                    self._last_worker_counts = tick_input.worker_counts
                # Drive ticks through the builtin orchestrator EngineProtocol.
                effects = await engine.tick(next_tick, tick_input)
                await self._apply_effects(effects)
                # Phase 1: reconcile per-GPU power-limit annotations, throttled
                # in steady state and forced for one interval after a scale-up.
                # Runs independently of diagnostic emission — annotations are a
                # cluster-side effect, not an operator-diagnostics signal.
                if self._should_sweep_power_annotations(now, effects):
                    await self._apply_power_annotations()
                emit_diagnostics = self._should_emit_tick_diagnostics(
                    next_tick, effects
                )
                if emit_diagnostics:
                    self._report_diagnostics(next_tick, effects.diagnostics)
                    self._log_decision_summary(effects)

                if self._recorder.enabled and emit_diagnostics:
                    try:
                        self._recorder.record(
                            tick_input,
                            effects,
                            self._last_metrics,
                            self._cumulative_gpu_hours,
                        )
                        if self._recorder.should_generate_report(tick_input.now_s):
                            self._recorder.generate_report()
                    except Exception as e:
                        logger.error(f"Diagnostics report failed: {e}")

                assert effects.next_tick is not None
                next_tick = effects.next_tick
        finally:
            self._recorder.finalize()
            if self._dashboard_runner is not None:
                await self._dashboard_runner.cleanup()
            if self._engine is not None:
                await self._engine.shutdown()


# ------------------------------------------------------------------
# Shared utility
# ------------------------------------------------------------------


def _log_fpm(wid: str, dp: int, fpm: ForwardPassMetrics, label: str) -> None:
    sched = fpm.scheduled_requests
    queued = fpm.queued_requests
    logger.info(
        f"FPM {label} engine {wid}:dp{dp}: "
        f"wall_time={fpm.wall_time:.4f}s, "
        f"sched(prefill_tok={sched.sum_prefill_tokens}, "
        f"prefill_req={sched.num_prefill_requests}, "
        f"decode_kv={sched.sum_decode_kv_tokens}, "
        f"decode_req={sched.num_decode_requests}), "
        f"queued(prefill_tok={queued.sum_prefill_tokens}, "
        f"decode_kv={queued.sum_decode_kv_tokens})"
    )
