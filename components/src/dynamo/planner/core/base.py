# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime I/O plumbing for the native planner.

This module contains **zero decision logic**.  It only gathers data from the
outside world (Prometheus, FPM subscribers, K8s connectors) and applies
scaling decisions back.  All scaling logic lives in
:class:`~dynamo.planner.core.state_machine.PlannerStateMachine`.

Subclasses (PrefillPlanner, DecodePlanner, AggPlanner, DisaggPlanner) set
mode-specific flags and override ``_bootstrap_regression`` and
``_apply_effects``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Optional, Union

import aiohttp.web
from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException
from prometheus_client import start_http_server

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY, _initialize_gpu_counts
from dynamo.planner.core.state_machine import PlannerStateMachine
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
from dynamo.planner.errors import PlannerError
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.live_dashboard import start_live_dashboard
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.planner.monitoring.traffic_metrics import Metrics, PrometheusAPIClient
from dynamo.planner.monitoring.worker_info import WorkerInfo, resolve_worker_info
from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
    from dynamo.llm import FpmEventSubscriber
    from dynamo.planner.monitoring.aic_power_optimizer import AICPowerOptimizer, PowerAwareConfig

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
            ssl_verify=config.metric_pulling_prometheus_ssl_verify,
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

        # Diagnostics recorder
        self._recorder = DiagnosticsRecorder(config=config)

        # Live dashboard runner (started in _async_init)
        self._dashboard_runner: Optional[aiohttp.web.AppRunner] = None

        # AIC closed-loop optimizer (Phase 3; None when disabled or not configured).
        self._aic_optimizer: Optional["AICPowerOptimizer"] = None

        # State machine (created after WorkerInfo is resolved)
        self._state_machine: Optional[PlannerStateMachine] = None

    # ------------------------------------------------------------------
    # State machine access
    # ------------------------------------------------------------------

    def _ensure_state_machine(self) -> PlannerStateMachine:
        if self._state_machine is None:
            caps = build_worker_capabilities(
                self.config,
                self.prefill_worker_info,
                self.decode_worker_info,
            )
            self._state_machine = PlannerStateMachine(self.config, caps)
            self._warm_predictors()
        return self._state_machine

    @property
    def state_machine(self) -> PlannerStateMachine:
        return self._ensure_state_machine()

    def _warm_predictors(self) -> None:
        if self.config.load_predictor_warmup_trace is None:
            return
        assert self._state_machine is not None
        try:
            metrics = extract_metrics_from_mooncake(
                self.config.load_predictor_warmup_trace,
                self.config.throughput_adjustment_interval_seconds,
            )
            self._state_machine.warm_load_predictors(
                [
                    TrafficObservation(
                        duration_s=self.config.throughput_adjustment_interval_seconds,
                        num_req=float(m["request_count"]),
                        isl=float(m["avg_isl"]),
                        osl=float(m["avg_osl"]),
                    )
                    for m in metrics
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to warm load predictors: {e}")

    # ------------------------------------------------------------------
    # Async init
    # ------------------------------------------------------------------

    async def _async_init(self) -> None:
        if hasattr(self, "connector") and hasattr(self.connector, "_async_init"):
            await self.connector._async_init()

        defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)
        logger.info("Validating deployment...")
        await self.connector.validate_deployment(
            prefill_component_name=(
                defaults.prefill_worker_k8s_name
                if self.require_prefill and defaults
                else None
            ),
            decode_component_name=(
                defaults.decode_worker_k8s_name
                if self.require_decode and defaults
                else None
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

        # AIC optimizer startup sweep (Phase 3).
        if self.config.enable_aic_optimizer:
            try:
                from dynamo.planner.monitoring.aic_power_optimizer import AICPowerOptimizer

                self._aic_optimizer = AICPowerOptimizer(
                    self.config, self.prometheus_metrics
                )
                initial_config = await asyncio.to_thread(self._aic_optimizer.optimize)
                if initial_config is not None:
                    await self._apply_aic_config(initial_config)
            except Exception as exc:
                logger.error(
                    "AIC optimizer startup failed — planner will continue with "
                    "static power enforcement: %s",
                    exc,
                )
                self._aic_optimizer = None

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
            try:
                fresh = self.connector.get_worker_info(sub_type, self.config.backend)
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

        if changed and self._state_machine is not None:
            self._state_machine.update_capabilities(
                build_worker_capabilities(
                    self.config,
                    self.prefill_worker_info,
                    self.decode_worker_info,
                )
            )

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

        hit_rate_str = f"{m.kv_hit_rate:.3f}" if m.kv_hit_rate is not None else "n/a"
        logger.info(
            f"Observed num_req: {m.num_req:.2f} isl: {m.isl:.2f} osl: {m.osl:.2f} "
            f"kv_hit_rate: {hit_rate_str}"
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

        # Aggregate token throughput for AIC drift detection (§5.6).
        rps = m.num_req / max(1.0, self.config.throughput_adjustment_interval)
        total_tokens_per_s = rps * (m.isl + m.osl) if m.isl > 0 and m.osl > 0 else None

        return TrafficObservation(
            duration_s=self.config.throughput_adjustment_interval_seconds,
            num_req=m.num_req,
            isl=m.isl,
            osl=m.osl,
            kv_hit_rate=m.kv_hit_rate,
            # AIC optimizer signals (Phase 3) — latency in seconds to match
            # PrometheusAPIClient conventions; m.ttft / m.itl are in ms.
            ttft_avg=m.ttft / 1000.0 if m.ttft and m.ttft > 0 else None,
            itl_avg=m.itl / 1000.0 if m.itl and m.itl > 0 else None,
            total_tokens_per_s=total_tokens_per_s,
        )

    async def _collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        """Pull only the KV hit rate from Prometheus over ``duration_s``.

        Used in load-only deployments: the load tick only needs the hit rate
        to discount prefill work, so we skip the five other (unused) traffic
        queries to keep the per-load-tick scrape cheap.

        Returns ``None`` when the router metric is unavailable (e.g.
        Prometheus source is "frontend"); the state machine treats that as
        a no-discount fallback.
        """
        assert self.model_name is not None
        if duration_s <= 0:
            return None
        interval_str = f"{int(duration_s)}s"
        hit_rate = self.prometheus_traffic_client.get_avg_kv_hit_rate(
            interval_str, self.model_name
        )
        # Mirror the observed value into Metrics so the diagnostics recorder
        # sees the up-to-date hit rate even on load-only ticks.
        self._last_metrics.kv_hit_rate = hit_rate
        hit_rate_str = f"{hit_rate:.3f}" if hit_rate is not None else "n/a"
        logger.info(f"Observed kv_hit_rate over {interval_str}: {hit_rate_str}")
        if hit_rate is None:
            return None
        return TrafficObservation(
            duration_s=duration_s,
            num_req=0.0,
            isl=0.0,
            osl=0.0,
            kv_hit_rate=hit_rate,
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
            if tick.run_throughput_scaling:
                traffic = await self._collect_traffic()
            else:
                traffic = await self._collect_kv_hit_rate_observation(
                    tick.traffic_metrics_duration_s
                )
        if tick.need_worker_states:
            worker_counts = await self._collect_worker_counts()
        if tick.need_worker_fpm:
            fpm_obs = self._collect_fpm()

        # Populate FPM-sourced scheduled token counts on the traffic observation
        # so AICPowerOptimizer.update_correction() can gate per-side EMA updates
        # on actual scheduled work (prevents idle-side coefficient drag — §5.3).
        if traffic is not None and fpm_obs is not None:
            if fpm_obs.prefill:
                traffic.scheduled_prefill_tokens = float(
                    sum(
                        fpm.scheduled_requests.sum_prefill_tokens
                        for fpm in fpm_obs.prefill.values()
                    )
                )
            if fpm_obs.decode:
                traffic.scheduled_decode_kv_tokens = float(
                    sum(
                        fpm.scheduled_requests.sum_decode_kv_tokens
                        for fpm in fpm_obs.decode.values()
                    )
                )

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
        K8s is the source of truth — no local cache.

        Called after _apply_effects() on every tick so newly-created pods
        pick up the cap within one tick, and AIC-driven cap changes (Phase 3+)
        propagate to all running pods automatically.
        """
        if not self.config.enable_power_awareness:
            return
        if not isinstance(self.connector, KubernetesConnector):
            return

        pods_and_limits: list[tuple] = []
        if self.require_prefill:
            for pod in self.connector.get_component_pods(SubComponentType.PREFILL):
                pods_and_limits.append(
                    (pod, str(self.config.prefill_engine_gpu_power_limit))
                )
        if self.require_decode:
            for pod in self.connector.get_component_pods(SubComponentType.DECODE):
                pods_and_limits.append(
                    (pod, str(self.config.decode_engine_gpu_power_limit))
                )

        for pod, limit_str in pods_and_limits:
            current = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
            if current == limit_str:
                continue
            try:
                self.connector.kube_api.patch_pod_annotation(
                    pod.metadata.name, POWER_ANNOTATION_KEY, limit_str
                )
                logger.info(
                    "Annotated pod %s with %s=%s",
                    pod.metadata.name,
                    POWER_ANNOTATION_KEY,
                    limit_str,
                )
            except Exception as e:
                logger.warning(
                    "Failed to patch power annotation on pod %s: %s",
                    pod.metadata.name,
                    e,
                )

    def _publish_power_budget_metrics(
        self, num_p: int, num_d: int
    ) -> None:
        """Emit power budget gauges (Phase 1, dashboard observability only).

        Uses static config values — not DCGM — so budget enforcement in
        _apply_power_budget() is unaffected when DCGM goes down.
        """
        if self.prometheus_port == 0 or not self.config.enable_power_awareness:
            return
        if self.config.total_gpu_power_limit is None:
            return

        p_gpu = self.config.prefill_engine_num_gpu or 0
        d_gpu = self.config.decode_engine_num_gpu or 0
        projected = (
            num_p * self.config.prefill_engine_gpu_power_limit * p_gpu
            + num_d * self.config.decode_engine_gpu_power_limit * d_gpu
        )
        budget = self.config.total_gpu_power_limit
        pm = self.prometheus_metrics
        pm.power_budget_total_watts.set(budget)
        pm.power_projected_watts.set(projected)
        pm.power_budget_utilization.set(projected / budget if budget > 0 else 0.0)

    def _min_prefill_max_num_batched_tokens(self) -> Optional[int]:
        """Return the minimum ``max_num_batched_tokens`` reported by prefill workers.

        Used to derive a defense-in-depth absolute prefill admission threshold
        (§5.7 B6).  The minimum-across-workers is conservative: a single
        under-reporting worker does not silently raise the effective floor.
        Returns ``None`` when no prefill worker has synced this value via MDC.
        """
        val = self.prefill_worker_info.max_num_batched_tokens if self.require_prefill else None
        return val if val and val > 0 else None

    async def _apply_aic_config(self, new_config: "PowerAwareConfig") -> None:
        """Apply an AIC-recommended config to the planner state (§6.4 + §6.7).

        Order of operations:
        1. Update per-GPU power caps in PlannerConfig (picked up by
           _apply_power_annotations() on the next tick).
        2. Request replica scaling via the connector.
        3. Update drift-comparison reference in the optimizer.
        4. Derive and emit implied admission thresholds (inherit + autoset).
        5. Fan out /busy_threshold POSTs to all frontend pods (autoset only).
        """
        if new_config is None:
            return

        # 1. Update config caps — _apply_power_annotations() reads these.
        self.config.prefill_engine_gpu_power_limit = new_config.cap_p
        self.config.decode_engine_gpu_power_limit = new_config.cap_d
        logger.info(
            "_apply_aic_config: cap_p=%dW cap_d=%dW n_p=%d n_d=%d",
            new_config.cap_p,
            new_config.cap_d,
            new_config.n_p,
            new_config.n_d,
        )

        # 2. Request replica scaling.
        targets: list[TargetReplica] = []
        if self.require_prefill and new_config.n_p > 0:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    desired_replicas=new_config.n_p,
                    component_name=self.prefill_worker_info.k8s_name,
                )
            )
        if self.require_decode and new_config.n_d > 0:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    desired_replicas=new_config.n_d,
                    component_name=self.decode_worker_info.k8s_name,
                )
            )
        if targets:
            await self._apply_scaling_targets(targets, blocking=False)

        # 3. Pin drift-comparison reference (§5.6).
        if self._aic_optimizer is not None:
            self._aic_optimizer._estimated_throughput = (
                new_config.aic_seq_per_s_per_replica
                * new_config.n_d
                * (new_config.isl + new_config.osl)
            )

        # 4. Admission gauges (inherit + autoset both record implied θ).
        if self.prometheus_port != 0 and self.config.admission_mode != "off":
            pm = self.prometheus_metrics
            pm.admission_implied_theta_decode.set(new_config.theta_decode_impl)
            pm.admission_implied_theta_prefill_frac.set(new_config.theta_prefill_frac_impl)

        # 5. Fan out POSTs to frontends (autoset only).
        if self.config.admission_mode == "autoset" and isinstance(
            self.connector, KubernetesConnector
        ):
            await self._fanout_busy_threshold(new_config)

    async def _fanout_busy_threshold(self, new_config: "PowerAwareConfig") -> None:
        """POST /busy_threshold to every frontend replica (§5.7 autoset path)."""
        margin = self.config.admission_safety_margin
        theta_d_set = min(1.0, margin * new_config.theta_decode_impl)
        theta_pf_set = min(1.0, margin * new_config.theta_prefill_frac_impl)

        # B6: absolute prefill threshold (defense-in-depth vs MDC DEFAULT_MAX_TOKENS).
        min_M = self._min_prefill_max_num_batched_tokens()
        theta_p_abs_set: Optional[int]
        if min_M is not None:
            theta_p_abs_set = math.ceil(theta_pf_set * min_M)
            if self.prometheus_port != 0:
                self.prometheus_metrics.admission_set_theta_prefill_abs.set(theta_p_abs_set)
        else:
            theta_p_abs_set = None
            if self.prometheus_port != 0:
                self.prometheus_metrics.admission_max_batched_tokens_unavailable_total.inc()
            logger.critical(
                "No prefill worker reports max_num_batched_tokens via MDC; "
                "absolute prefill admission threshold cannot be derived. "
                "Frontend's DEFAULT_MAX_TOKENS=10M fallback may silently disable "
                "the fractional prefill check until MDC re-syncs. "
                "Investigate worker MDC plumbing."
            )

        if self.prometheus_port != 0:
            pm = self.prometheus_metrics
            pm.admission_set_theta_decode.set(theta_d_set)
            pm.admission_set_theta_prefill_frac.set(theta_pf_set)

        assert isinstance(self.connector, KubernetesConnector)
        pods = self.connector.list_frontend_pods()
        if not pods:
            logger.debug("No frontend pods found; skipping /busy_threshold fanout.")
            return

        model_name = self.model_name or self.config.model_name or ""
        # Per-pod port resolution: operator emits a named ``http`` port on
        # the frontend container, so we read each pod's actual port from its
        # spec and only fall back to the (deprecated) ``frontend_http_port``
        # config field for legacy manifests.  This honors per-DGD port
        # overrides without a config mirror — closes design-doc OQ #13.
        results = await asyncio.gather(
            *(
                self.connector.post_busy_threshold(
                    pod,
                    model_name,
                    port=self.connector.resolve_frontend_http_port(
                        pod, fallback=self.config.frontend_http_port
                    ),
                    active_decode_blocks_threshold=theta_d_set,
                    active_prefill_tokens_threshold=theta_p_abs_set,
                    active_prefill_tokens_threshold_frac=theta_pf_set,
                )
                for pod in pods
            ),
            return_exceptions=True,
        )

        failed = [
            p.metadata.name
            for p, r in zip(pods, results)
            if isinstance(r, Exception)
        ]
        if failed:
            if self.prometheus_port != 0:
                self.prometheus_metrics.admission_partial_success_total.inc(len(failed))
            logger.error(
                "Failed to POST /busy_threshold to %d/%d frontend pods: %s",
                len(failed),
                len(pods),
                failed,
            )

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
        """Log a one-line summary of the scaling decision after each tick."""
        decision = effects.scale_to
        diag = effects.diagnostics

        sm = self.state_machine
        current_p = sm._num_p_workers
        current_d = sm._num_d_workers

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

    def _report_diagnostics(self, tick: ScheduledTick, diag: TickDiagnostics) -> None:
        if self.prometheus_port == 0:
            return
        pm = self.prometheus_metrics
        interval = self.config.throughput_adjustment_interval_seconds

        pm.estimated_ttft_ms.set(diag.estimated_ttft_ms or 0)
        pm.estimated_itl_ms.set(diag.estimated_itl_ms or 0)

        pm.predicted_requests_per_second.set(
            diag.predicted_num_req / interval
            if diag.predicted_num_req is not None and interval > 0
            else 0
        )
        pm.predicted_input_sequence_tokens.set(diag.predicted_isl or 0)
        pm.predicted_output_sequence_tokens.set(diag.predicted_osl or 0)

        pm.engine_prefill_capacity_requests_per_second.set(diag.engine_rps_prefill or 0)
        pm.engine_decode_capacity_requests_per_second.set(diag.engine_rps_decode or 0)

        if tick.run_load_scaling:
            pm.load_scaling_decision.state(diag.load_decision_reason or "unset")
        if tick.run_throughput_scaling:
            pm.throughput_scaling_decision.state(
                diag.throughput_decision_reason or "unset"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        next_tick = self.state_machine.initial_tick(time.time())
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
                effects = self.state_machine.on_tick(next_tick, tick_input)
                await self._apply_effects(effects)
                await self._apply_power_annotations()  # Phase 1: annotate pods with GPU power limit

                # Phase 3: AIC closed-loop optimizer update + conditional re-sweep.
                if self._aic_optimizer is not None and tick_input.traffic is not None:
                    interval_str = f"{self.config.throughput_adjustment_interval}s"
                    # DGD name lives on the connector, not PlannerConfig.  When
                    # running outside K8s (virtual / global-planner connector),
                    # the DCGM selector is meaningless and the queries below
                    # gracefully degrade to None.
                    dgd_name = (
                        self.connector.graph_deployment_name
                        if isinstance(self.connector, KubernetesConnector)
                        else ""
                    )
                    # DCGM attribution labels carry the K8s namespace
                    # (exported_namespace), not the dynamo logical namespace.
                    # Resolve via the connector when running in-cluster.
                    k8s_ns = (
                        self.connector.kube_api.current_namespace
                        if isinstance(self.connector, KubernetesConnector)
                        else ""
                    )

                    if self.config.mode == "disagg":
                        obs_ttft = self.prometheus_traffic_client.get_avg_time_to_first_token(
                            interval_str, self.model_name or ""
                        )
                        obs_itl = self.prometheus_traffic_client.get_avg_inter_token_latency(
                            interval_str, self.model_name or ""
                        )
                        self._aic_optimizer.update_correction(
                            traffic=tick_input.traffic,
                            observed_ttft_avg=obs_ttft if obs_ttft else None,
                            observed_itl_avg=obs_itl if obs_itl else None,
                            observed_power_w_prefill=(
                                self.prometheus_traffic_client.get_avg_per_gpu_power_by_component(
                                    interval=interval_str,
                                    k8s_namespace=k8s_ns,
                                    dgd_name=dgd_name,
                                    component="prefill",
                                    service_key=self.prefill_worker_info.k8s_name or "",
                                )
                                if self.require_prefill
                                else None
                            ),
                            observed_power_w_decode=(
                                self.prometheus_traffic_client.get_avg_per_gpu_power_by_component(
                                    interval=interval_str,
                                    k8s_namespace=k8s_ns,
                                    dgd_name=dgd_name,
                                    component="decode",
                                    service_key=self.decode_worker_info.k8s_name or "",
                                )
                                if self.require_decode
                                else None
                            ),
                        )
                    else:  # agg mode
                        obs_ttft = self.prometheus_traffic_client.get_avg_time_to_first_token(
                            interval_str, self.model_name or ""
                        )
                        obs_itl = self.prometheus_traffic_client.get_avg_inter_token_latency(
                            interval_str, self.model_name or ""
                        )
                        self._aic_optimizer.update_correction(
                            traffic=tick_input.traffic,
                            observed_ttft_avg=obs_ttft if obs_ttft else None,
                            observed_itl_avg=obs_itl if obs_itl else None,
                            observed_power_w_agg=(
                                self.prometheus_traffic_client.get_avg_per_gpu_power_by_component(
                                    interval=interval_str,
                                    k8s_namespace=k8s_ns,
                                    dgd_name=dgd_name,
                                    component="agg",
                                    service_key=self.decode_worker_info.k8s_name or "",
                                )
                                if self.require_decode
                                else None
                            ),
                        )

                    if self._aic_optimizer.should_reoptimize(tick_input.traffic):
                        logger.info(
                            "AIC optimizer: drift detected — triggering re-sweep."
                        )
                        try:
                            new_aic_config = await asyncio.to_thread(
                                self._aic_optimizer.optimize
                            )
                            if new_aic_config is not None:
                                await self._apply_aic_config(new_aic_config)
                        except Exception as exc:
                            logger.error(
                                "AIC re-sweep exception (planner continues with "
                                "current config): %s",
                                exc,
                            )

                self._report_diagnostics(next_tick, effects.diagnostics)
                self._log_decision_summary(effects)

                if self._recorder.enabled:
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
