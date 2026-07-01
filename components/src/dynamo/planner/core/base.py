# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed planner runtime plumbing.

This is the candidate replacement for ``core/base.py``.  Planner decision logic
still lives in the engine/plugins; this class owns engine lifecycle,
diagnostics, tick orchestration, and delegates deployment/metrics I/O to a
``PlannerEnvironment``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import aiohttp.web
from prometheus_client import start_http_server

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core import util
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
from dynamo.planner.environment.interface import PlannerEnvironment
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.live_dashboard import start_live_dashboard
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake
from dynamo.runtime import DistributedRuntime

if TYPE_CHECKING:
    from dynamo.planner.monitoring.worker_info import WorkerInfo

logger = logging.getLogger(__name__)


def _engine_caps(worker_info, num_gpu: Optional[int]) -> Optional[EngineCapabilities]:
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


def build_worker_capabilities(state: DeploymentState) -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=_engine_caps(state.prefill.info, state.prefill.num_gpus),
        decode=_engine_caps(state.decode.info, state.decode.num_gpus),
    )


class NativePlannerBase:
    """Base adapter shared by planner modes."""

    require_prefill: bool = False
    require_decode: bool = False

    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        config: PlannerConfig,
        environment: PlannerEnvironment,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.environment = environment
        self.namespace = config.namespace

        self.prometheus_port = config.metric_reporting_prometheus_port
        self.prometheus_metrics = PlannerPrometheusMetrics()
        if self.prometheus_port != 0:
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    "Started Prometheus metrics server on port %s",
                    self.prometheus_port,
                )
            except Exception as exc:
                logger.error("Failed to start Prometheus metrics server: %s", exc)
            self.prometheus_metrics.sla_target_ttft_ms.set(config.ttft_ms)
            self.prometheus_metrics.sla_target_itl_ms.set(config.itl_ms)

        self._cumulative_gpu_hours: float = 0.0
        self._last_gpu_hours_update_ts: Optional[float] = None
        self._recorder = DiagnosticsRecorder(config=config)
        self._dashboard_runner: Optional[aiohttp.web.AppRunner] = None
        self._engine: Optional[EngineProtocol] = None
        self._last_worker_counts: Optional[WorkerCounts] = None

    async def _async_init(self) -> None:
        await self.environment.initialize()

        await self._bootstrap_regression()
        await self._bootstrap_engine_plugins_if_needed()

        if self.config.advisory:
            logger.info(
                "[ADVISORY] Planner started in advisory mode; "
                "scaling decisions will be logged but NOT executed."
            )

        if self.config.live_dashboard_port:
            try:
                self._dashboard_runner = await start_live_dashboard(
                    self._recorder, self.config.live_dashboard_port
                )
            except Exception as exc:
                logger.error("Failed to start live dashboard: %s", exc)

    def _build_worker_capabilities(self) -> WorkerCapabilities:
        return build_worker_capabilities(self.environment.deployment_state())

    def _runtime_namespace(self) -> str:
        return self.environment.runtime_namespace()

    def _required_worker_info(self, component: SubComponentType) -> "WorkerInfo":
        state = self.environment.deployment_state()
        info = (
            state.prefill.info
            if component == SubComponentType.PREFILL
            else state.decode.info
        )
        if info is None:
            raise RuntimeError(f"Missing worker info for {component.value}")
        return info

    def _ensure_engine(self) -> EngineProtocol:
        if self._engine is not None:
            return self._engine
        # Deliberately lazy: importing planner core should not load the plugin stack.
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )
        from dynamo.planner.plugins.builtins.observe import EnvironmentObservePlugin

        self._engine = OrchestratorEngineAdapter(
            self.config,
            self._build_worker_capabilities(),
            observe_plugin=EnvironmentObservePlugin(
                self.environment,
                require_prefill=self.require_prefill,
                require_decode=self.require_decode,
            ),
        )
        return self._engine

    async def _install_benchmark_fpms(
        self,
        *,
        prefill_fpms=None,
        decode_fpms=None,
        agg_fpms=None,
    ) -> None:
        # Keep the orchestrator dependency aligned with lazy engine construction.
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
        except Exception as exc:
            logger.warning("Failed to warm load predictors: %s", exc)
            return None

    async def _bootstrap_engine_plugins_if_needed(self) -> None:
        # Keep the orchestrator dependency aligned with lazy engine construction.
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

    async def _bootstrap_regression(self) -> None:
        pass

    async def _refresh_and_update_capabilities(self) -> None:
        old_state = self.environment.deployment_state().clone()
        await self.environment.refresh()
        new_state = self.environment.deployment_state().clone()
        if not util.deployment_state_changed(
            old_state,
            new_state,
            self.require_prefill,
            self.require_decode,
        ):
            return
        update_engine_capabilities = getattr(self._engine, "update_capabilities", None)
        if callable(update_engine_capabilities):
            update_engine_capabilities(self._build_worker_capabilities())

    async def _collect_traffic(self) -> Optional[TrafficObservation]:
        return await self.environment.collect_traffic()

    async def _collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        return await self.environment.collect_kv_hit_rate_observation(duration_s)

    def _collect_fpm(self) -> FpmObservations:
        return self.environment.collect_fpm()

    def _emit_observed_traffic_metrics(self) -> None:
        if self.prometheus_port == 0:
            return
        m = self.environment.metrics_state()
        if not m.is_valid():
            return
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
        state = self.environment.deployment_state()
        return WorkerCounts(
            ready_num_prefill=(
                state.prefill.replicas.active if self.require_prefill else None
            ),
            ready_num_decode=(
                state.decode.replicas.active if self.require_decode else None
            ),
            expected_num_prefill=(
                state.prefill.replicas.expected if self.require_prefill else None
            ),
            expected_num_decode=(
                state.decode.replicas.expected if self.require_decode else None
            ),
            prefill_scaling_in_progress=(
                self.require_prefill and state.prefill.replicas.scaling
            ),
            decode_scaling_in_progress=(
                self.require_decode and state.decode.replicas.scaling
            ),
        )

    async def _gather_tick_input(self, tick: ScheduledTick) -> TickInput:
        now = time.time()
        traffic = None
        worker_counts = None
        fpm_obs = None

        if tick.need_traffic_metrics:
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

    async def _observe_tick(
        self, engine: EngineProtocol, tick: ScheduledTick
    ) -> TickInput:
        observe = getattr(engine, "observe", None)
        if callable(observe):
            return await observe(tick, time.time())
        return await self._gather_tick_input(tick)

    def _publish_observation_metrics(self, tick_input: TickInput) -> None:
        if tick_input.traffic is not None:
            self._emit_observed_traffic_metrics()
        if tick_input.fpm_observations is not None and self.prometheus_port != 0:
            self._emit_per_engine_fpm(
                tick_input.fpm_observations.prefill,
                tick_input.fpm_observations.decode,
            )

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        pass

    async def _apply_scaling_targets(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        if self.config.advisory or not targets:
            return
        await self.environment.apply_scaling(targets, blocking=blocking)

    def _log_decision_summary(self, effects: PlannerEffects) -> None:
        decision = effects.scale_to
        diag = effects.diagnostics

        if self._last_worker_counts is not None:
            current_p = self._last_worker_counts.ready_num_prefill or 0
            current_d = self._last_worker_counts.ready_num_decode or 0
        else:
            current_p = 0
            current_d = 0

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

    def _publish_inventory_and_gpu_hours(self, tick_input: TickInput) -> None:
        if tick_input.worker_counts is None:
            return
        num_p = tick_input.worker_counts.ready_num_prefill or 0
        num_d = tick_input.worker_counts.ready_num_decode or 0

        now = tick_input.now_s
        state = self.environment.deployment_state()
        prefill_gpus = state.prefill.num_gpus or 0
        decode_gpus = state.decode.num_gpus or 0
        if self._last_gpu_hours_update_ts is not None:
            dt_s = max(0.0, now - self._last_gpu_hours_update_ts)
            self._cumulative_gpu_hours += (
                (num_p * prefill_gpus + num_d * decode_gpus) * dt_s / 3600.0
            )
        self._last_gpu_hours_update_ts = now

        if self.prometheus_port == 0:
            return
        self.prometheus_metrics.num_prefill_replicas.set(num_p)
        self.prometheus_metrics.num_decode_replicas.set(num_d)
        self.prometheus_metrics.gpu_hours.set(self._cumulative_gpu_hours)

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
        diag = effects.diagnostics
        return (
            tick.run_load_scaling
            or tick.run_throughput_scaling
            or effects.scale_to is not None
            or bool(diag.audit_events)
            or bool(diag.short_circuit_reason)
        )

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

                await self._refresh_and_update_capabilities()
                tick_input = await self._observe_tick(engine, next_tick)
                self._publish_observation_metrics(tick_input)
                self._publish_inventory_and_gpu_hours(tick_input)
                if tick_input.worker_counts is not None:
                    self._last_worker_counts = tick_input.worker_counts

                effects = await engine.tick(next_tick, tick_input)
                await self._apply_effects(effects)
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
                            self.environment.metrics_state(),
                            self._cumulative_gpu_hours,
                        )
                        if self._recorder.should_generate_report(tick_input.now_s):
                            self._recorder.generate_report()
                    except Exception as exc:
                        logger.error("Diagnostics report failed: %s", exc)

                assert effects.next_tick is not None
                next_tick = effects.next_tick
        finally:
            self._recorder.finalize()
            await self.environment.shutdown()
            if self._dashboard_runner is not None:
                await self._dashboard_runner.cleanup()
            if self._engine is not None:
                await self._engine.shutdown()
