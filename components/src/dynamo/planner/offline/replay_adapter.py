# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter that drives the planner core via the PlannerReplayBridge.

The bridge (Rust, PyO3) runs the offline simulation step-by-step.
This adapter sits between the bridge and the planner tick engine:

    Bridge.advance_to(tick_ms) -> raw metrics dict
    Adapter._build_tick_input() -> TickInput
    EngineProtocol.tick() -> PlannerEffects
    Adapter -> Bridge.apply_scaling(prefill, decode)

The tick engine is selected by ``config.scheduling.use_orchestrator``:

- ``False`` (default): legacy PSM path — ``PlannerStateMachine`` +
  ``_PSMEngineAdapter``. Byte-for-byte identical to pre-PR-8 replay.
- ``True``: orchestrator path — ``OrchestratorEngineAdapter`` wrapping
  ``LocalPlannerOrchestrator`` + the 5 builtin plugins. Produces the
  same ``PlannerEffects.scale_to`` / ``next_tick`` as PSM (dual-path
  parity test lock) with plugin-era observability (Prometheus metrics,
  audit events, plugin-aware diagnostics).

Replay keeps its sync ``run()`` API on both paths; async calls on the
orchestrator path (``bootstrap_from_fpms`` / ``tick``) run inside a
single replay-scoped event loop so callers don't need to change.

Supports both aggregated and disaggregated topologies. No I/O, no
runtime dependencies. Fully deterministic with offline replay.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.engine_protocol import EngineProtocol, _PSMEngineAdapter
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
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
from dynamo.planner.monitoring.traffic_metrics import Metrics

logger = logging.getLogger(__name__)


@dataclass
class ScalingEvent:
    """Record of a single scaling decision."""

    at_s: float
    component: str  # "agg", "prefill", or "decode"
    from_count: int
    to_count: int
    reason: Optional[str] = None


@dataclass
class ReplayPlannerReport:
    """Enriched report combining trace metrics and planner diagnostics."""

    trace_report: dict[str, Any]
    scaling_events: list[ScalingEvent] = field(default_factory=list)
    diagnostics_log: list[TickDiagnostics] = field(default_factory=list)
    total_ticks: int = 0
    html_report_path: Optional[str] = None


def _build_fpm_from_dict(d: dict[str, Any]) -> ForwardPassMetrics:
    """Convert a bridge FPM snapshot dict into a ForwardPassMetrics struct."""
    return ForwardPassMetrics(
        worker_id=str(d["worker_id"]),
        dp_rank=0,
        wall_time=d["wall_time"],
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=d["num_prefill_requests"],
            sum_prefill_tokens=d["sum_prefill_tokens"],
            var_prefill_length=d["var_prefill_length"],
            sum_prefill_kv_tokens=d["sum_prefill_kv_tokens"],
            num_decode_requests=d["num_decode_requests"],
            sum_decode_kv_tokens=d["sum_decode_kv_tokens"],
            var_decode_kv_tokens=d["var_decode_kv_tokens"],
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=d["num_queued_prefill"],
            sum_prefill_tokens=d["sum_queued_prefill_tokens"],
            var_prefill_length=d["var_queued_prefill_length"],
            num_decode_requests=d["num_queued_decode"],
            sum_decode_kv_tokens=d["sum_queued_decode_kv_tokens"],
            var_decode_kv_tokens=d["var_queued_decode_kv_tokens"],
        ),
    )


def _update_fpm_cache(
    cache: dict[tuple[str, int], ForwardPassMetrics],
    snapshots: list[dict[str, Any]],
    active_count: int,
) -> None:
    """Update a last-seen FPM cache with new snapshots and prune removed workers."""
    for snap in snapshots:
        fpm = _build_fpm_from_dict(snap)
        cache[(fpm.worker_id, fpm.dp_rank)] = fpm

    # Prune cache down to active_count entries. Workers are removed
    # highest-ID-first during scale-down, so keep the lowest IDs.
    while len(cache) > active_count:
        # Remove the highest worker ID entry
        worst_key = max(cache.keys(), key=lambda k: int(k[0]))
        del cache[worst_key]


class ReplayPlannerAdapter:
    """Drives the planner state machine using the PlannerReplayBridge.

    Supports both ``mode="agg"`` and ``mode="disagg"``.
    """

    def __init__(
        self,
        planner_config: PlannerConfig,
        bridge: Any,  # PlannerReplayBridge (Rust pyclass)
        capabilities: Optional[WorkerCapabilities] = None,
        warmup_observations: Optional[list[TrafficObservation]] = None,
    ) -> None:
        self._config = planner_config
        self._bridge = bridge
        self._capabilities = capabilities
        self._is_disagg = planner_config.mode == "disagg"

        # Tick engine selected by the feature flag. On PSM path
        # ``self._sm`` is the actual state machine (reused for helpers
        # like ``warm_load_predictors``). On orchestrator path it is
        # ``None``; a throwaway PSM inside ``OrchestratorEngineAdapter.
        # bootstrap_from_fpms`` handles regression bootstrap instead.
        use_orchestrator = planner_config.scheduling.use_orchestrator
        self._use_orchestrator = use_orchestrator
        self._sm: Optional[PlannerStateMachine] = None
        self._engine: EngineProtocol
        if use_orchestrator:
            from dynamo.planner.plugins.orchestrator.engine_adapter import (
                OrchestratorEngineAdapter,
            )

            self._engine = OrchestratorEngineAdapter(
                planner_config, capabilities or WorkerCapabilities()
            )
            # Replay's ``run()`` is synchronous; we own a scoped event
            # loop to drive the async engine calls without forcing
            # callers to use ``asyncio.run``.
            self._loop = asyncio.new_event_loop()
        else:
            self._sm = PlannerStateMachine(planner_config, capabilities)
            self._engine = _PSMEngineAdapter(self._sm)
            self._loop = None  # type: ignore[assignment]

        # Last-seen FPM caches (separate for prefill/decode)
        self._prefill_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        self._decode_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}

        # Scaling targets -- used as `expected` in WorkerCounts
        self._scaling_target_prefill: Optional[int] = None
        self._scaling_target_decode: Optional[int] = None

        # Diagnostics recorder for HTML report generation
        decode_max_kv = (
            capabilities.decode.max_kv_tokens
            if capabilities and capabilities.decode
            else None
        )
        self._recorder = DiagnosticsRecorder(
            config=planner_config, max_kv_tokens=decode_max_kv
        )
        self._cumulative_gpu_hours: float = 0.0
        self._last_tick_s: float = 0.0
        self._last_traffic: Metrics = Metrics()

        # Warmup path: PSM exposes ``warm_load_predictors`` directly; on
        # the orchestrator path we route the same list through
        # ``bootstrap_plugins(historical_traffic=...)`` which primes the
        # builtin predictor identically.
        if warmup_observations:
            if self._use_orchestrator:
                self._run_sync(
                    self._engine.bootstrap_plugins(  # type: ignore[attr-defined]
                        historical_traffic=warmup_observations
                    )
                )
            else:
                assert self._sm is not None
                self._sm.warm_load_predictors(warmup_observations)

    # ------------------------------------------------------------------
    # Sync/async bridging
    # ------------------------------------------------------------------

    def _run_sync(self, coro):
        """Run a coroutine on the replay-owned event loop. Used to call
        the orchestrator path's async APIs from replay's sync surface."""
        assert self._loop is not None, "sync bridge only available on orchestrator path"
        return self._loop.run_until_complete(coro)

    def run(self) -> ReplayPlannerReport:
        """Run the full replay with planner-in-the-loop."""
        next_tick = self._engine.initial_tick(0.0)
        scaling_events: list[ScalingEvent] = []
        diagnostics_log: list[TickDiagnostics] = []
        total_ticks = 0

        while True:
            tick_ms = next_tick.at_s * 1000.0
            result = self._bridge.advance_to(tick_ms)

            if result["is_done"]:
                break

            tick_input = self._build_tick_input(next_tick, result)
            # ``EngineProtocol.tick`` is async. On PSM path the
            # ``_PSMEngineAdapter`` wraps PSM's sync ``on_tick`` in an
            # async-defined-but-never-awaits shim, so ``run_until_complete``
            # returns synchronously without yielding to the loop. On
            # orchestrator path it genuinely awaits the pipeline.
            if self._use_orchestrator:
                effects: PlannerEffects = self._run_sync(
                    self._engine.tick(next_tick, tick_input)
                )
            else:
                # Fast path for PSM: skip the event-loop roundtrip.
                assert self._sm is not None
                effects = self._sm.on_tick(next_tick, tick_input)
            diagnostics_log.append(effects.diagnostics)
            total_ticks += 1

            # Update GPU-hours and record diagnostics snapshot
            self._record_diagnostics(tick_input, effects, result)

            # Clear scaling targets once active counts match
            active_p = result["active_prefill_count"]
            active_d = result["active_decode_count"]
            if (
                self._scaling_target_prefill is not None
                and active_p == self._scaling_target_prefill
            ):
                self._scaling_target_prefill = None
            if (
                self._scaling_target_decode is not None
                and active_d == self._scaling_target_decode
            ):
                self._scaling_target_decode = None

            if effects.scale_to is not None:
                self._apply_scaling(effects, result, tick_input.now_s, scaling_events)

            if effects.next_tick is None:
                break
            next_tick = effects.next_tick

        trace_report = self._bridge.finalize()
        html_report_path = self._recorder.finalize()
        return ReplayPlannerReport(
            trace_report=trace_report,
            scaling_events=scaling_events,
            diagnostics_log=diagnostics_log,
            total_ticks=total_ticks,
            html_report_path=html_report_path,
        )

    def _record_diagnostics(
        self,
        tick_input: TickInput,
        effects: PlannerEffects,
        result: dict[str, Any],
    ) -> None:
        """Update GPU-hours tracking and feed the diagnostics recorder."""
        if not self._recorder.enabled:
            return

        now_s = tick_input.now_s
        if self._last_tick_s > 0.0:
            dt_h = (now_s - self._last_tick_s) / 3600.0
            num_p = result["active_prefill_count"]
            num_d = result["active_decode_count"]
            gpu_p = self._config.prefill_engine_num_gpu or 0
            gpu_d = self._config.decode_engine_num_gpu or 0
            self._cumulative_gpu_hours += (num_p * gpu_p + num_d * gpu_d) * dt_h
        self._last_tick_s = now_s

        self._recorder.record(
            tick_input,
            effects,
            self._last_traffic,
            self._cumulative_gpu_hours,
        )

    def _apply_scaling(
        self,
        effects: PlannerEffects,
        result: dict[str, Any],
        now_s: float,
        scaling_events: list[ScalingEvent],
    ) -> None:
        """Apply scaling decisions and record events."""
        scale = effects.scale_to
        assert scale is not None
        current_p = result["active_prefill_count"]
        current_d = result["active_decode_count"]
        target_p = scale.num_prefill if scale.num_prefill is not None else current_p
        target_d = scale.num_decode if scale.num_decode is not None else current_d

        if target_p == current_p and target_d == current_d:
            return

        self._bridge.apply_scaling(target_p, target_d)

        if self._is_disagg:
            if scale.num_prefill is not None and target_p != current_p:
                direction = "scale_up" if target_p > current_p else "scale_down"
                logger.info(
                    "Planner scaling prefill: %d -> %d at t=%.1fs (%s)",
                    current_p,
                    target_p,
                    now_s,
                    direction,
                )
                self._scaling_target_prefill = target_p
                scaling_events.append(
                    ScalingEvent(
                        at_s=now_s,
                        component="prefill",
                        from_count=current_p,
                        to_count=target_p,
                        reason=direction,
                    )
                )
            if scale.num_decode is not None and target_d != current_d:
                direction = "scale_up" if target_d > current_d else "scale_down"
                logger.info(
                    "Planner scaling decode: %d -> %d at t=%.1fs (%s)",
                    current_d,
                    target_d,
                    now_s,
                    direction,
                )
                self._scaling_target_decode = target_d
                scaling_events.append(
                    ScalingEvent(
                        at_s=now_s,
                        component="decode",
                        from_count=current_d,
                        to_count=target_d,
                        reason=direction,
                    )
                )
        else:
            direction = "scale_up" if target_d > current_d else "scale_down"
            logger.info(
                "Planner scaling: %d -> %d workers at t=%.1fs (%s)",
                current_d,
                target_d,
                now_s,
                direction,
            )
            self._scaling_target_decode = target_d
            scaling_events.append(
                ScalingEvent(
                    at_s=now_s,
                    component="agg",
                    from_count=current_d,
                    to_count=target_d,
                    reason=direction,
                )
            )

    def _feed_extra_fpm_to_regression(
        self,
        decode_snaps: list[dict[str, Any]],
        prefill_snaps: list[dict[str, Any]],
    ) -> None:
        """Feed accumulated FPM snapshots to regression, excluding the last
        per worker (which will be added by _observe_fpm via fpm_observations).
        This avoids double-counting the cached snapshot.

        Works on both paths via ``_get_regression(kind)`` so
        orchestrator replay and PSM replay share identical snapshot
        feeding. Returns early on easy mode (no regressions) or when
        the requested regression slot isn't installed (the install gap
        is fixed via the empty-regression bootstrap in
        ``_install_benchmark_fpms``).
        """
        if self._is_easy_mode():
            return  # easy mode has no regression models

        if self._config.mode == "agg":
            agg_reg = self._get_regression("agg")
            if agg_reg is None:
                return
            last_idx_per_worker: dict[int, int] = {}
            for i, snap in enumerate(decode_snaps):
                last_idx_per_worker[snap["worker_id"]] = i
            exclude = set(last_idx_per_worker.values())
            for i, snap in enumerate(decode_snaps):
                if i in exclude:
                    continue
                fpm = _build_fpm_from_dict(snap)
                if fpm.wall_time > 0.0:
                    agg_reg.add_observation(fpm)
        else:
            has_prefill = self._config.mode in ("prefill", "disagg")
            has_decode = self._config.mode in ("decode", "disagg")
            if has_prefill:
                p_reg = self._get_regression("prefill")
                if p_reg is not None:
                    last_idx: dict[int, int] = {}
                    for i, snap in enumerate(prefill_snaps):
                        last_idx[snap["worker_id"]] = i
                    exclude = set(last_idx.values())
                    for i, snap in enumerate(prefill_snaps):
                        if i in exclude:
                            continue
                        fpm = _build_fpm_from_dict(snap)
                        if fpm.wall_time > 0.0:
                            p_reg.add_observation(fpm)
            if has_decode:
                d_reg = self._get_regression("decode")
                if d_reg is not None:
                    last_idx = {}
                    for i, snap in enumerate(decode_snaps):
                        last_idx[snap["worker_id"]] = i
                    exclude = set(last_idx.values())
                    for i, snap in enumerate(decode_snaps):
                        if i in exclude:
                            continue
                        fpm = _build_fpm_from_dict(snap)
                        if fpm.wall_time > 0.0:
                            d_reg.add_observation(fpm)

    def _is_easy_mode(self) -> bool:
        """Easy-mode check routed via config — both paths honour this
        the same way (no regression in non-SLA modes)."""
        return self._config.optimization_target != "sla"

    def _get_regression(self, kind: str):
        """Return the regression model for ``kind`` (``"agg"`` /
        ``"prefill"`` / ``"decode"``) regardless of engine path.

        PSM path: read directly from ``self._sm.{_agg,_prefill,_decode}_regression``.
        Orchestrator path: read from the orchestrator's shared store
        (populated by ``bootstrap_from_fpms`` → ``install_regressions``).
        """
        if self._use_orchestrator:
            # The adapter hides the orchestrator; access via its public
            # bootstrap hook doesn't help — read through the underlying
            # orchestrator attribute we know is there.
            orch = getattr(self._engine, "_orchestrator", None)
            if orch is None:
                return None
            return orch.get_regression(kind)
        if self._sm is None:
            return None
        attr = f"_{kind}_regression"
        return getattr(self._sm, attr, None)

    def _build_tick_input(
        self, tick: ScheduledTick, result: dict[str, Any]
    ) -> TickInput:
        """Convert bridge result dict to planner TickInput."""
        now_s = result["now_ms"] / 1000.0

        worker_counts = None
        if tick.need_worker_states:
            active_p = result["active_prefill_count"]
            active_d = result["active_decode_count"]
            expected_p = (
                self._scaling_target_prefill
                if self._scaling_target_prefill is not None
                else active_p
            )
            expected_d = (
                self._scaling_target_decode
                if self._scaling_target_decode is not None
                else active_d
            )
            worker_counts = WorkerCounts(
                ready_num_prefill=active_p if self._is_disagg else None,
                ready_num_decode=active_d,
                expected_num_prefill=expected_p if self._is_disagg else None,
                expected_num_decode=expected_d,
            )

        fpm_observations = None
        if tick.need_worker_fpm:
            prefill_snaps = result.get("prefill_fpm_snapshots", [])
            decode_snaps = result.get("decode_fpm_snapshots", [])

            _update_fpm_cache(
                self._prefill_fpm_cache, prefill_snaps, result["active_prefill_count"]
            )
            _update_fpm_cache(
                self._decode_fpm_cache, decode_snaps, result["active_decode_count"]
            )

            # In offline replay, we accumulate many FPM snapshots per tick
            # (one per engine pass). Feed ALL non-idle snapshots directly to
            # the regression models for a representative fit. The last-per-worker
            # cache is only used for the FpmObservations dict (worker count
            # reconciliation), not as the sole regression input.
            prefill_dict = (
                dict(self._prefill_fpm_cache) if self._prefill_fpm_cache else None
            )
            decode_dict = (
                dict(self._decode_fpm_cache) if self._decode_fpm_cache else None
            )
            self._feed_extra_fpm_to_regression(decode_snaps, prefill_snaps)
            fpm_observations = FpmObservations(
                prefill=prefill_dict,
                decode=decode_dict,
            )

        traffic = None
        if tick.need_traffic_metrics:
            t = self._bridge.drain_traffic()
            duration_s = t.get("duration_s", 0.0)
            if duration_s > 0:
                num_req = float(t.get("num_req", 0))
                # The mocker publishes avg_kv_hit_rate as 0.0 when the
                # window had no admissions with non-zero ISL blocks;
                # pass it through as-is so the state machine can decide
                # whether to feed its predictor.
                traffic = TrafficObservation(
                    duration_s=duration_s,
                    num_req=num_req,
                    isl=t.get("avg_isl", 0.0),
                    osl=t.get("avg_osl", 0.0),
                    kv_hit_rate=t.get("avg_kv_hit_rate"),
                )
                # Stash observed TTFT/ITL for the diagnostics recorder.
                # When num_req == 0, the Rust accumulator returns 0 as a
                # placeholder; only record latency values when we actually
                # observed requests in this window.
                self._last_traffic = Metrics(
                    ttft=t.get("avg_ttft_ms") if num_req > 0 else None,
                    itl=t.get("avg_itl_ms") if num_req > 0 else None,
                    num_req=traffic.num_req,
                    isl=traffic.isl,
                    osl=traffic.osl,
                    kv_hit_rate=traffic.kv_hit_rate,
                )

        return TickInput(
            now_s=now_s,
            traffic=traffic,
            worker_counts=worker_counts,
            fpm_observations=fpm_observations,
        )
