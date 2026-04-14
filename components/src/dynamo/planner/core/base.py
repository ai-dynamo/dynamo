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
import time
from typing import TYPE_CHECKING, Optional, Union

from prometheus_client import start_http_server

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.core.budget import _initialize_gpu_counts
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
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
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
        max_num_batched_tokens=worker_info.max_num_batched_tokens
        if worker_info
        else None,
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
        self.model_name: Optional[str] = None

        # Connector
        self.connector: ConnectorType
        if not config.no_operation:
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

        # Diagnostics recorder
        self._recorder = DiagnosticsRecorder(config=config)

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
                self.config.throughput_adjustment_interval,
            )
            self._state_machine.warm_load_predictors(
                [
                    TrafficObservation(
                        duration_s=self.config.throughput_adjustment_interval,
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

        if not self.config.no_operation:
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

        await self._init_worker_info()

        if self.runtime is not None:
            if self.require_prefill:
                await self._init_fpm_subscriber("prefill")
            if self.require_decode:
                await self._init_fpm_subscriber("decode")

        await self._bootstrap_regression()

    async def _init_worker_info(self) -> None:
        connector = getattr(self, "connector", None)
        self.prefill_worker_info, self.decode_worker_info = resolve_worker_info(
            backend=self.config.backend,
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
            connector=connector,
            config_model_name=getattr(self.config, "model_name", ""),
            no_operation=self.config.no_operation,
        )
        self.model_name = (
            self.decode_worker_info.model_name or self.prefill_worker_info.model_name
        )

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
            f"{self.namespace}.{worker_info.component_name}.{worker_info.endpoint}"
        )
        sub = FpmEventSubscriber(endpoint)
        sub.start_tracking()
        logger.info(
            f"FPM tracker started for {worker_info.component_name}.{worker_info.endpoint}"
        )

        if component == "prefill":
            self._prefill_fpm_sub = sub
        else:
            self._decode_fpm_sub = sub

    async def _bootstrap_regression(self) -> None:
        """Override in subclasses to bootstrap regression models."""
        pass

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
            f"{self.namespace}.{component_name}.{endpoint_name}"
        ).client()
        await asyncio.sleep(0.1)
        return client

    async def _get_worker_counts_raw(self) -> tuple[int, int, bool]:
        """Returns (num_prefill, num_decode, is_stable) from connector or runtime."""
        if hasattr(self, "connector") and isinstance(
            self.connector, KubernetesConnector
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
        """Pull traffic metrics from Prometheus."""
        num_p, num_d, _ = await self._get_worker_counts_raw()

        if self.prometheus_port != 0:
            self.prometheus_metrics.num_prefill_replicas.set(num_p)
            self.prometheus_metrics.num_decode_replicas.set(num_d)
            gpu_hours = (
                (
                    num_p * (self.config.prefill_engine_num_gpu or 0)
                    + num_d * (self.config.decode_engine_num_gpu or 0)
                )
                * self.config.throughput_adjustment_interval
                / 3600
            )
            self._cumulative_gpu_hours += gpu_hours
            self.prometheus_metrics.gpu_hours.set(self._cumulative_gpu_hours)

        assert self.model_name is not None
        interval_str = f"{self.config.throughput_adjustment_interval}s"
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

        logger.info(
            f"Observed num_req: {m.num_req:.2f} isl: {m.isl:.2f} osl: {m.osl:.2f}"
        )

        if self.prometheus_port != 0:
            self.prometheus_metrics.observed_ttft_ms.set(m.ttft)
            self.prometheus_metrics.observed_itl_ms.set(m.itl)
            self.prometheus_metrics.observed_requests_per_second.set(
                m.num_req / self.config.throughput_adjustment_interval
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
            duration_s=self.config.throughput_adjustment_interval,
            num_req=m.num_req,
            isl=m.isl,
            osl=m.osl,
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
        return WorkerCounts(
            ready_num_prefill=num_p if self.require_prefill else None,
            ready_num_decode=num_d if self.require_decode else None,
            expected_num_prefill=(num_p if is_stable else None)
            if self.require_prefill
            else None,
            expected_num_decode=(num_d if is_stable else None)
            if self.require_decode
            else None,
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
            traffic = await self._collect_traffic()
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

    async def _apply_scaling_targets(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        """Shared helper: send scaling targets to connector."""
        if self.config.no_operation or not targets:
            return
        await self.connector.set_component_replicas(targets, blocking=blocking)

    # ------------------------------------------------------------------
    # Diagnostics reporting (shared across all adapters)
    # ------------------------------------------------------------------

    def _report_diagnostics(self, diag: TickDiagnostics) -> None:
        if self.prometheus_port == 0:
            return
        pm = self.prometheus_metrics
        interval = self.config.throughput_adjustment_interval

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

        pm.load_scaling_decision.state(diag.load_decision_reason or "unset")
        pm.throughput_scaling_decision.state(diag.throughput_decision_reason or "unset")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        next_tick = self.state_machine.initial_tick(time.time())
        poll_interval = self.config.load_adjustment_interval / 10

        while True:
            now = time.time()
            if now < next_tick.at_s:
                await asyncio.sleep(min(next_tick.at_s - now, poll_interval))
                continue

            tick_input = await self._gather_tick_input(next_tick)
            effects = self.state_machine.on_tick(next_tick, tick_input)
            await self._apply_effects(effects)
            self._report_diagnostics(effects.diagnostics)

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
