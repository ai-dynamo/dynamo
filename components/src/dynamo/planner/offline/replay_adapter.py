# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter that drives the planner core from the unified replay loop.

The Rust offline simulation (``PlannerReplayBridge.run``) owns the drive
loop and calls back into this adapter once per ``PlannerTick`` event, so
the adapter is a callback hook rather than an external stepper:

    Bridge.run(adapter)                        # Rust owns the loop
      adapter.initial_tick_ms()      -> first tick time
      per PlannerTick:
        adapter.on_tick(metrics)     -> _build_tick_input() -> TickInput
                                        EngineProtocol.tick() -> PlannerEffects
                                        -> {target_prefill, target_decode, next_tick_ms}
        # Rust applies the scaling decision and re-arms the next tick itself
      adapter.finalize(trace_report) -> ReplayPlannerReport

The tick engine is the builtin orchestrator path:
``OrchestratorEngineAdapter`` wrapping ``LocalPlannerOrchestrator`` +
the builtin local-planner plugins. It preserves the planner's
``PlannerEffects.scale_to`` replay contract while using plugin-aware
observability (Prometheus metrics, audit events, diagnostics).

The simulation steps itself — replay no longer drives the bridge
externally. Async orchestrator calls (``bootstrap_from_fpms`` / ``tick``)
run inside a single replay-scoped event loop so callers don't change.

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
from dynamo.planner.core.engine_protocol import EngineProtocol
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
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

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
        dp_rank=int(d.get("dp_rank", 0)),
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

    # Prune by logical worker, retaining every DP-rank entry for each active
    # worker. Workers are removed highest-ID-first during scale-down.
    active_worker_ids = set(
        sorted({worker_id for worker_id, _ in cache}, key=int)[:active_count]
    )
    for key in list(cache):
        if key[0] not in active_worker_ids:
            del cache[key]


def _merge_traffic(
    acc: Optional[dict[str, Any]], window: dict[str, Any]
) -> dict[str, Any]:
    """Merge two TrafficStats dicts into one window.

    Exact for every field the planner's scaling consumes:
      - ``duration_s``/``num_req``: summed.
      - ``avg_isl``/``avg_osl``: num_req-weighted — their denominator *is*
        ``num_req``, so a num_req-weighted mean of per-window means re-sums to
        the exact overall mean.
      - ``avg_kv_hit_rate``: weighted by ``hit_rate_count`` (its true
        denominator: router admissions with ``isl_blocks > 0``), so the merge
        reconstructs the exact sample mean rather than approximating it.
      - ``avg_accept_length``: weighted by ``accept_length_forward_count``
        (decode request-forwards, its true denominator), exact across windows.

    ``avg_ttft_ms``/``avg_itl_ms`` are num_req-weighted approximations (their
    per-sample counts are not carried across windows); they feed diagnostics
    only, never the scaling trajectory."""
    if acc is None:
        return dict(window)
    na = float(acc.get("num_req", 0.0))
    nw = float(window.get("num_req", 0.0))
    n = na + nw

    def _weighted(key: str, wa: float, ww: float) -> float:
        w = wa + ww
        if w <= 0:
            return 0.0
        return (acc.get(key, 0.0) * wa + window.get(key, 0.0) * ww) / w

    hit_a = float(acc.get("hit_rate_count", 0.0))
    hit_w = float(window.get("hit_rate_count", 0.0))
    fwd_a = float(acc.get("accept_length_forward_count", 0.0))
    fwd_w = float(window.get("accept_length_forward_count", 0.0))

    merged: dict[str, Any] = {
        "duration_s": acc.get("duration_s", 0.0) + window.get("duration_s", 0.0),
        "num_req": n,
        # Carry the native denominators so chained multi-window merges stay exact.
        "hit_rate_count": hit_a + hit_w,
        "accept_length_forward_count": fwd_a + fwd_w,
        # num_req-weighted: exact for isl/osl, diagnostics-only for ttft/itl.
        "avg_isl": _weighted("avg_isl", na, nw),
        "avg_osl": _weighted("avg_osl", na, nw),
        "avg_ttft_ms": _weighted("avg_ttft_ms", na, nw),
        "avg_itl_ms": _weighted("avg_itl_ms", na, nw),
        # Count-weighted by the true denominator -> exact across windows.
        "avg_kv_hit_rate": _weighted("avg_kv_hit_rate", hit_a, hit_w),
    }
    a_acc = acc.get("avg_accept_length")
    a_win = window.get("avg_accept_length")
    if a_acc is None and a_win is None:
        merged["avg_accept_length"] = None
    elif a_acc is None:
        merged["avg_accept_length"] = a_win
    elif a_win is None:
        merged["avg_accept_length"] = a_acc
    else:
        fwd = fwd_a + fwd_w
        merged["avg_accept_length"] = (
            (a_acc * fwd_a + a_win * fwd_w) / fwd if fwd > 0 else None
        )
    return merged


class ReplayPlannerAdapter:
    """Drives the plugin planner using the PlannerReplayBridge.

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

        self._engine: EngineProtocol
        self._warmup_observations = list(warmup_observations or [])
        self._orchestrator_bootstrapped = False
        # Inject a ``VirtualClock`` so plugin scheduler / circuit breaker /
        # HOLD_LAST cache see trace time, not real wall-clock.
        self._engine = OrchestratorEngineAdapter(
            planner_config,
            capabilities or WorkerCapabilities(),
            clock=VirtualClock(),
        )
        # Replay's ``run()`` is synchronous; we own a scoped event loop to
        # drive the async engine calls without forcing callers to use
        # ``asyncio.run``.
        self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.new_event_loop()

        # Last-seen FPM caches (separate for prefill/decode)
        self._prefill_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        self._decode_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        self._pending_prefill_fpm_snaps: list[dict[str, Any]] = []
        self._pending_decode_fpm_snaps: list[dict[str, Any]] = []
        # Partial traffic window accumulated across ticks until a throughput tick
        # consumes it (``None`` = nothing pending).
        self._pending_traffic: Optional[dict[str, Any]] = None

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

        # Orchestrator Bootstrap is deferred until ``run()`` because replay
        # installs benchmark FPMs after adapter construction and before the
        # first tick.

    # ------------------------------------------------------------------
    # Sync/async bridging
    # ------------------------------------------------------------------

    def _run_sync(self, coro):
        """Run a coroutine on the replay-owned event loop. Used to call
        the orchestrator path's async APIs from replay's sync surface."""
        assert self._loop is not None, "sync bridge only available on orchestrator path"
        return self._loop.run_until_complete(coro)

    def _bootstrap_orchestrator_if_needed(self) -> None:
        if self._orchestrator_bootstrapped:
            return
        self._run_sync(
            self._engine.bootstrap_plugins(  # type: ignore[attr-defined]
                historical_traffic=self._warmup_observations or None
            )
        )
        self._orchestrator_bootstrapped = True

    def install_benchmark_fpms(
        self,
        *,
        prefill_fpms: Optional[list[ForwardPassMetrics]] = None,
        decode_fpms: Optional[list[ForwardPassMetrics]] = None,
        agg_fpms: Optional[list[ForwardPassMetrics]] = None,
    ) -> None:
        """Install AIC benchmark FPMs into the regression model(s).

        Normal replay uses ``OrchestratorEngineAdapter
        .install_regressions_from_fpms``.

        Without this, replay's throughput regression stays empty and
        planner-in-the-loop scaling decisions diverge from live planner
        behavior."""
        self._engine.install_regressions_from_fpms(  # type: ignore[attr-defined]
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
            agg_fpms=agg_fpms,
        )

    # ------------------------------------------------------------------
    # Inverted drive: the Rust ``PlannerReplayBridge.run(self)`` owns the loop and
    # calls ``initial_tick_ms`` once then ``on_tick`` per ``PlannerTick`` event. The
    # entrypoint wraps the returned trace_report via ``finalize``. (Replaces the old
    # Python while-loop that drove ``bridge.advance_to`` + ``bridge.apply_scaling``.)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bootstrap the orchestrator and compute the first tick. Idempotent."""
        self._bootstrap_orchestrator_if_needed()
        self._pending_tick: ScheduledTick = self._engine.initial_tick(0.0)
        self._scaling_events: list[ScalingEvent] = []
        self._diagnostics_log: list[TickDiagnostics] = []
        self._total_ticks = 0

    def initial_tick_ms(self) -> float:
        """First tick time in milliseconds (called by the Rust PlannerHook)."""
        if not self._orchestrator_bootstrapped or not hasattr(self, "_pending_tick"):
            self.start()
        return self._pending_tick.at_s * 1000.0

    def on_tick(self, result: dict[str, Any]) -> dict[str, Any]:
        """Drive one planner tick from the bridge's metrics dict. Returns the scaling
        decision (absolute targets, ``None`` = unchanged) + the next tick time in ms
        (``None`` = stop) for the Rust loop to apply and re-arm."""
        tick = self._pending_tick
        tick_input = self._build_tick_input(tick, result)
        effects: PlannerEffects = self._run_sync(self._engine.tick(tick, tick_input))
        emit_diagnostics = self._should_emit_tick_diagnostics(tick, effects)
        if emit_diagnostics:
            self._diagnostics_log.append(effects.diagnostics)
        self._total_ticks += 1
        self._record_diagnostics(tick_input, effects, result, emit_diagnostics)

        # Clear scaling targets once active counts match.
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

        target_prefill: Optional[int] = None
        target_decode: Optional[int] = None
        if effects.scale_to is not None:
            target_prefill, target_decode = self._compute_scale_decision(
                effects, result, tick_input.now_s
            )

        next_tick_ms: Optional[float] = None
        if effects.next_tick is not None:
            self._pending_tick = effects.next_tick
            next_tick_ms = effects.next_tick.at_s * 1000.0

        return {
            "target_prefill": target_prefill,
            "target_decode": target_decode,
            "next_tick_ms": next_tick_ms,
        }

    def finalize(self, trace_report: dict[str, Any]) -> ReplayPlannerReport:
        """Assemble the enriched report from accumulated planner state. Called by the
        entrypoint after the Rust ``run()`` returns the trace_report dict."""
        try:
            html_report_path = self._recorder.finalize()
            return ReplayPlannerReport(
                trace_report=trace_report,
                scaling_events=self._scaling_events,
                diagnostics_log=self._diagnostics_log,
                total_ticks=self._total_ticks,
                html_report_path=html_report_path,
            )
        finally:
            self.close()

    def close(self) -> None:
        """Shut down the engine and the replay-scoped event loop. Idempotent so it
        runs cleanly from both ``finalize`` (success) and the entrypoint's error
        path (when ``bridge.run`` raises before ``finalize`` is reached)."""
        loop = self._loop
        if loop is None:
            return
        self._loop = None
        try:
            loop.run_until_complete(self._engine.shutdown())
        finally:
            loop.close()

    def _record_diagnostics(
        self,
        tick_input: TickInput,
        effects: PlannerEffects,
        result: dict[str, Any],
        emit_diagnostics: bool,
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

        if not emit_diagnostics:
            return

        self._recorder.record(
            tick_input,
            effects,
            self._last_traffic,
            self._cumulative_gpu_hours,
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

    def _compute_scale_decision(
        self,
        effects: PlannerEffects,
        result: dict[str, Any],
        now_s: float,
    ) -> tuple[Optional[int], Optional[int]]:
        """Compute the (prefill, decode) absolute scale targets and record the scaling
        event. Returns ``(None, None)`` for a no-op. The Rust loop applies the targets,
        so this no longer calls ``bridge.apply_scaling``."""
        scale = effects.scale_to
        if scale is None:
            raise ValueError(
                "_compute_scale_decision requires effects.scale_to to be set"
            )
        current_p = result["active_prefill_count"]
        current_d = result["active_decode_count"]
        target_p = scale.num_prefill if scale.num_prefill is not None else current_p
        target_d = scale.num_decode if scale.num_decode is not None else current_d

        if target_p == current_p and target_d == current_d:
            return (None, None)

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
                self._scaling_events.append(
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
                self._scaling_events.append(
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
            self._scaling_events.append(
                ScalingEvent(
                    at_s=now_s,
                    component="agg",
                    from_count=current_d,
                    to_count=target_d,
                    reason=direction,
                )
            )
        return (target_p, target_d)

    def _feed_extra_fpm_to_regression(
        self,
        decode_snaps: list[dict[str, Any]],
        prefill_snaps: list[dict[str, Any]],
    ) -> None:
        """Feed accumulated FPM snapshots to regression, excluding the last
        per worker (which will be added by _observe_fpm via fpm_observations).
        This avoids double-counting the cached snapshot.

        Works via ``_get_regression(kind)`` for orchestrator replay. Returns
        early on easy mode (no regressions) or when the requested regression
        slot isn't installed.
        """
        if self._is_easy_mode():
            return  # easy mode has no regression models

        if self._config.mode == "agg":
            agg_reg = self._get_regression("agg")
            if agg_reg is None:
                return
            last_idx_per_worker: dict[tuple[Any, int], int] = {}
            for i, snap in enumerate(decode_snaps):
                last_idx_per_worker[
                    (snap["worker_id"], int(snap.get("dp_rank", 0)))
                ] = i
            exclude = set(last_idx_per_worker.values())
            for i, snap in enumerate(decode_snaps):
                if i in exclude:
                    continue
                fpm = _build_fpm_from_dict(snap)
                if fpm.wall_time > 0.0:
                    agg_reg.add_observations({(fpm.worker_id, fpm.dp_rank): fpm})
        else:
            has_prefill = self._config.mode in ("prefill", "disagg")
            has_decode = self._config.mode in ("decode", "disagg")
            if has_prefill:
                p_reg = self._get_regression("prefill")
                if p_reg is not None:
                    last_idx: dict[tuple[Any, int], int] = {}
                    for i, snap in enumerate(prefill_snaps):
                        last_idx[(snap["worker_id"], int(snap.get("dp_rank", 0)))] = i
                    exclude = set(last_idx.values())
                    for i, snap in enumerate(prefill_snaps):
                        if i in exclude:
                            continue
                        fpm = _build_fpm_from_dict(snap)
                        if fpm.wall_time > 0.0:
                            p_reg.add_observations({(fpm.worker_id, fpm.dp_rank): fpm})
            if has_decode:
                d_reg = self._get_regression("decode")
                if d_reg is not None:
                    last_idx = {}
                    for i, snap in enumerate(decode_snaps):
                        last_idx[(snap["worker_id"], int(snap.get("dp_rank", 0)))] = i
                    exclude = set(last_idx.values())
                    for i, snap in enumerate(decode_snaps):
                        if i in exclude:
                            continue
                        fpm = _build_fpm_from_dict(snap)
                        if fpm.wall_time > 0.0:
                            d_reg.add_observations({(fpm.worker_id, fpm.dp_rank): fpm})

    def _is_easy_mode(self) -> bool:
        """Easy-mode check routed via config — both paths honour this
        the same way (no regression in non-SLA modes)."""
        return self._config.optimization_target != "sla"

    def _get_regression(self, kind: str):
        """Return the regression model for ``kind`` (``"agg"`` /
        ``"prefill"`` / ``"decode"``).

        Normal path: read from the orchestrator adapter's shared scaling
        state. That state owns both benchmark-installed regressions and the
        empty live-regression slots that replay must feed when no AIC data was
        installed.
        """
        attr = f"_{kind}_regression"
        state = getattr(self._engine, "_scaling_state", None)
        if state is not None:
            regression = getattr(state, attr, None)
            if regression is not None:
                return regression
        # Benchmark regressions are also published to the orchestrator so
        # external plugins can read them during bootstrap hooks.
        orch = getattr(self._engine, "_orchestrator", None)
        if orch is None:
            return None
        return orch.get_regression(kind)

    def _build_tick_input(
        self, tick: ScheduledTick, result: dict[str, Any]
    ) -> TickInput:
        """Convert bridge result dict to planner TickInput."""
        # Keep planner cadence on the scheduled replay clock. The Rust bridge
        # also advances idle gaps to this timestamp so traffic windows drain
        # with the same duration the planner sees.
        now_s = tick.at_s

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
                prefill_scaling_in_progress=(
                    self._is_disagg
                    and self._scaling_target_prefill is not None
                    and self._scaling_target_prefill != active_p
                ),
                decode_scaling_in_progress=(
                    self._scaling_target_decode is not None
                    and self._scaling_target_decode != active_d
                ),
            )

        fpm_observations = None
        if not hasattr(self, "_pending_prefill_fpm_snaps"):
            self._pending_prefill_fpm_snaps = []
        if not hasattr(self, "_pending_decode_fpm_snaps"):
            self._pending_decode_fpm_snaps = []
        self._pending_prefill_fpm_snaps.extend(result.get("prefill_fpm_snapshots", []))
        self._pending_decode_fpm_snaps.extend(result.get("decode_fpm_snapshots", []))
        if tick.need_worker_fpm:
            prefill_snaps = self._pending_prefill_fpm_snaps
            decode_snaps = self._pending_decode_fpm_snaps
            self._pending_prefill_fpm_snaps = []
            self._pending_decode_fpm_snaps = []

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

        # The Rust bridge drains the per-tick traffic window into ``result["traffic"]``;
        # accumulate it so a need_traffic_metrics tick sees the full window since the
        # last consumed one (the planner consumes traffic only on throughput ticks).
        tick_traffic = result.get("traffic")
        if tick_traffic is not None:
            self._pending_traffic = _merge_traffic(
                getattr(self, "_pending_traffic", None), tick_traffic
            )

        traffic = None
        if tick.need_traffic_metrics:
            t = getattr(self, "_pending_traffic", None) or {}
            self._pending_traffic = None
            duration_s = t.get("duration_s", 0.0)
            if duration_s > 0:
                num_req = float(t.get("num_req", 0))
                # The mocker publishes avg_kv_hit_rate as 0.0 when the
                # window had no admissions with non-zero ISL blocks;
                # pass it through as-is so the planner can distinguish
                # "no datapoint" from an explicit zero hit rate.
                traffic = TrafficObservation(
                    duration_s=duration_s,
                    num_req=num_req,
                    isl=t.get("avg_isl", 0.0),
                    osl=t.get("avg_osl", 0.0),
                    kv_hit_rate=t.get("avg_kv_hit_rate"),
                    accept_length=t.get("avg_accept_length"),
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
                    accept_length=traffic.accept_length,
                )

        return TickInput(
            now_s=now_s,
            traffic=traffic,
            worker_counts=worker_counts,
            fpm_observations=fpm_observations,
        )
