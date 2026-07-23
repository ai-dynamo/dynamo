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
from dynamo.planner.plugins.clock import Clock, VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

logger = logging.getLogger(__name__)


@dataclass
class ScalingEvent:
    """Record of a committed scaling action."""

    at_s: float
    component: str  # "agg", "prefill", or "decode"
    from_count: int
    to_count: int
    reason: Optional[str] = None


@dataclass(frozen=True)
class ScalingProposal:
    """One local-planner proposal awaiting mediation and actuation.

    ``None`` retains the planner's component mask: it means the local planner
    expressed no opinion about that component. ``current_*`` is the ready count
    observed by the local planner when it produced the proposal and is used only
    as the legacy direct-path actuation baseline.
    """

    at_s: float
    current_prefill: int
    current_decode: int
    target_prefill: Optional[int] = None
    target_decode: Optional[int] = None


@dataclass(frozen=True)
class ScaleMediationEvent:
    """Mediation outcome for a local proposal.

    Mediation and actuation are deliberately separate: an approved proposal may
    be partially applied, while an adapter may receive a committed partner action
    even when it did not produce a proposal on that tick.
    """

    at_s: float
    status: str
    target_prefill: Optional[int] = None
    target_decode: Optional[int] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class PlannerTickResult:
    """Result of one local planner tick before any scaling is committed."""

    proposal: Optional[ScalingProposal]
    next_tick_ms: Optional[float]


@dataclass
class ReplayPlannerReport:
    """Enriched report combining trace metrics and planner diagnostics."""

    trace_report: dict[str, Any]
    scaling_events: list[ScalingEvent] = field(default_factory=list)
    diagnostics_log: list[TickDiagnostics] = field(default_factory=list)
    total_ticks: int = 0
    html_report_path: Optional[str] = None
    # Appended to preserve the legacy positional constructor order above.
    mediation_events: list[ScaleMediationEvent] = field(default_factory=list)


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
    active_worker_ids: list[int],
) -> None:
    """Update a last-seen FPM cache with new snapshots and prune removed workers."""
    for snap in snapshots:
        fpm = _build_fpm_from_dict(snap)
        cache[(fpm.worker_id, fpm.dp_rank)] = fpm

    active_worker_ids_as_str = {str(worker_id) for worker_id in active_worker_ids}
    for key in list(cache):
        if key[0] not in active_worker_ids_as_str:
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
        *,
        clock: Optional[Clock] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._config = planner_config
        self._bridge = bridge
        self._capabilities = capabilities or WorkerCapabilities()
        self._is_disagg = planner_config.mode == "disagg"

        self._engine: EngineProtocol
        self._warmup_observations = list(warmup_observations or [])
        self._orchestrator_bootstrapped = False
        # Inject a ``VirtualClock`` so plugin scheduler / circuit breaker /
        # HOLD_LAST cache see trace time, not real wall-clock.
        self._engine = OrchestratorEngineAdapter(
            planner_config,
            self._capabilities,
            clock=clock or VirtualClock(),
        )
        # Replay's ``run()`` is synchronous; we own a scoped event loop to
        # drive the async engine calls without forcing callers to use
        # ``asyncio.run``. A multi-deployment world injects its shared loop and
        # drives ``start_async`` / ``propose_tick_async`` directly.
        self._owns_loop = event_loop is None
        self._loop: Optional[asyncio.AbstractEventLoop] = (
            event_loop if event_loop is not None else asyncio.new_event_loop()
        )
        self._engine_shutdown = False

        # Last-seen FPM caches (separate for prefill/decode)
        self._prefill_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        self._decode_fpm_cache: dict[tuple[str, int], ForwardPassMetrics] = {}
        # Partial traffic window accumulated across ticks until a throughput tick
        # consumes it (``None`` = nothing pending).
        self._pending_traffic: Optional[dict[str, Any]] = None

        # Scaling targets -- used as `expected` in WorkerCounts
        self._scaling_target_prefill: Optional[int] = None
        self._scaling_target_decode: Optional[int] = None
        # Last runtime inventory observed by a tick. Total counts include workers
        # that are starting or draining when the bridge exposes them.
        self._last_ready_prefill: Optional[int] = None
        self._last_ready_decode: Optional[int] = None
        self._last_total_prefill: Optional[int] = None
        self._last_total_decode: Optional[int] = None
        self._has_total_prefill = False
        self._has_total_decode = False
        self._scaling_events: list[ScalingEvent] = []
        self._mediation_events: list[ScaleMediationEvent] = []
        self._diagnostics_log: list[TickDiagnostics] = []
        self._total_ticks = 0

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

    async def _bootstrap_orchestrator_if_needed_async(self) -> None:
        if self._orchestrator_bootstrapped:
            return
        await self._engine.bootstrap_plugins(  # type: ignore[attr-defined]
            historical_traffic=self._warmup_observations or None
        )
        self._orchestrator_bootstrapped = True

    def _bootstrap_orchestrator_if_needed(self) -> None:
        self._run_sync(self._bootstrap_orchestrator_if_needed_async())

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

    def _initialize_run_state(self) -> None:
        self._pending_tick: ScheduledTick = self._engine.initial_tick(0.0)
        self._scaling_events = []
        self._mediation_events = []
        self._diagnostics_log = []
        self._total_ticks = 0

    async def start_async(self) -> None:
        """Bootstrap on the caller's running loop and compute the first tick."""
        await self._bootstrap_orchestrator_if_needed_async()
        self._initialize_run_state()

    def start(self) -> None:
        """Bootstrap the orchestrator and compute the first tick. Idempotent."""
        self._run_sync(self.start_async())

    def initial_tick_ms(self) -> float:
        """First tick time in milliseconds (called by the Rust PlannerHook)."""
        return self._run_sync(self.initial_tick_ms_async())

    async def initial_tick_ms_async(self) -> float:
        """First tick time for a controller that owns the running event loop."""
        if not self._orchestrator_bootstrapped or not hasattr(self, "_pending_tick"):
            await self.start_async()
        return self._pending_tick.at_s * 1000.0

    async def propose_tick_async(self, result: dict[str, Any]) -> PlannerTickResult:
        """Run one local planner tick without committing its scale proposal.

        The caller owns mediation. It may reject the proposal, commit a partial
        target, or commit an action to a different adapter selected as a Global
        Planner partner. Planner cadence and diagnostics still advance normally.
        """
        tick = self._pending_tick
        tick_input = self._build_tick_input(tick, result)
        effects: PlannerEffects = await self._engine.tick(tick, tick_input)
        emit_diagnostics = self._should_emit_tick_diagnostics(tick, effects)
        if emit_diagnostics:
            self._diagnostics_log.append(effects.diagnostics)
        self._total_ticks += 1
        self._record_diagnostics(tick_input, effects, result, emit_diagnostics)

        proposal: Optional[ScalingProposal] = None
        if effects.scale_to is not None:
            proposal = self._build_scale_proposal(effects, result, tick_input.now_s)

        next_tick_ms: Optional[float] = None
        if effects.next_tick is not None:
            self._pending_tick = effects.next_tick
            next_tick_ms = effects.next_tick.at_s * 1000.0

        return PlannerTickResult(proposal=proposal, next_tick_ms=next_tick_ms)

    def propose_tick(self, result: dict[str, Any]) -> PlannerTickResult:
        """Synchronous wrapper used by the legacy single-deployment bridge."""
        return self._run_sync(self.propose_tick_async(result))

    def on_tick(self, result: dict[str, Any]) -> dict[str, Any]:
        """Legacy direct bridge path: propose, then immediately commit.

        ``PlannerReplayBridge`` historically treats every local proposal as an
        accepted action. Preserve that API while the multi-deployment world uses
        :meth:`propose_tick` plus explicit mediation/commit methods.
        """
        tick_result = self.propose_tick(result)
        proposal = tick_result.proposal
        if proposal is not None:
            self.observe_mediation(proposal, status="approved")
            self.observe_committed_scale(
                at_s=proposal.at_s,
                target_prefill=proposal.target_prefill,
                target_decode=proposal.target_decode,
                from_prefill=proposal.current_prefill,
                from_decode=proposal.current_decode,
            )

        return {
            "target_prefill": (
                proposal.target_prefill if proposal is not None else None
            ),
            "target_decode": proposal.target_decode if proposal is not None else None,
            "next_tick_ms": tick_result.next_tick_ms,
        }

    def build_report(self, trace_report: dict[str, Any]) -> ReplayPlannerReport:
        """Assemble a report without shutting down an injected shared loop."""
        html_report_path = self._recorder.finalize()
        return ReplayPlannerReport(
            trace_report=trace_report,
            scaling_events=self._scaling_events,
            mediation_events=self._mediation_events,
            diagnostics_log=self._diagnostics_log,
            total_ticks=self._total_ticks,
            html_report_path=html_report_path,
        )

    def finalize(self, trace_report: dict[str, Any]) -> ReplayPlannerReport:
        """Build the legacy report and close this adapter's planner resources."""
        try:
            report = self.build_report(trace_report)
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_exc:
                exc.add_note(f"Replay planner cleanup also failed: {cleanup_exc}")
            raise
        else:
            self.close()
            return report

    async def shutdown_async(self) -> None:
        """Shut down this planner engine on the caller's running event loop."""
        if self._engine_shutdown:
            return
        await self._engine.shutdown()
        self._engine_shutdown = True

    def close(self) -> None:
        """Shut down the engine and the replay-scoped event loop. Idempotent so it
        runs cleanly from both ``finalize`` (success) and the entrypoint's error
        path (when ``bridge.run`` raises before ``finalize`` is reached)."""
        loop = self._loop
        if loop is None:
            return
        if loop.is_running():
            raise RuntimeError(
                "ReplayPlannerAdapter.close() cannot run on an active event loop; "
                "await shutdown_async() from the shared-loop controller"
            )
        try:
            loop.run_until_complete(self.shutdown_async())
        finally:
            if self._owns_loop:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                finally:
                    loop.close()
                    self._loop = None

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
            gpu_p = (
                self._capabilities.prefill.num_gpu
                if self._capabilities.prefill is not None
                and self._capabilities.prefill.num_gpu is not None
                else 0
            )
            gpu_d = (
                self._capabilities.decode.num_gpu
                if self._capabilities.decode is not None
                and self._capabilities.decode.num_gpu is not None
                else 0
            )
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

    def _build_scale_proposal(
        self,
        effects: PlannerEffects,
        result: dict[str, Any],
        now_s: float,
    ) -> Optional[ScalingProposal]:
        """Build a masked local proposal without mutating actuation state."""
        scale = effects.scale_to
        if scale is None:
            raise ValueError(
                "_build_scale_proposal requires effects.scale_to to be set"
            )
        current_p = result["active_prefill_count"]
        current_d = result["active_decode_count"]
        target_p = scale.num_prefill if self._is_disagg else None
        target_d = scale.num_decode

        changed_p = target_p is not None and target_p != current_p
        changed_d = target_d is not None and target_d != current_d
        if not changed_p and not changed_d:
            return None

        return ScalingProposal(
            at_s=now_s,
            current_prefill=current_p,
            current_decode=current_d,
            target_prefill=target_p,
            target_decode=target_d,
        )

    def observe_mediation(
        self,
        proposal: ScalingProposal,
        *,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Record mediation without implying that any target was committed."""
        if status not in {"approved", "rejected", "error"}:
            raise ValueError(
                "mediation status must be 'approved', 'rejected', or 'error'"
            )
        self._mediation_events.append(
            ScaleMediationEvent(
                at_s=proposal.at_s,
                status=status,
                target_prefill=proposal.target_prefill,
                target_decode=proposal.target_decode,
                message=message,
            )
        )

    def observe_rejection(
        self, proposal: ScalingProposal, *, message: Optional[str] = None
    ) -> None:
        """Record a soft denial. Expected targets and scaling events stay unchanged."""
        self.observe_mediation(proposal, status="rejected", message=message)

    def observe_committed_scale(
        self,
        *,
        at_s: float,
        target_prefill: Optional[int] = None,
        target_decode: Optional[int] = None,
        from_prefill: Optional[int] = None,
        from_decode: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Observe targets actually committed to this deployment.

        This method is intentionally independent of :meth:`propose_tick`: a
        Global Planner may apply a cached or partially-consumed partner intent to
        an adapter that did not tick at ``at_s``. When no explicit ``from_*``
        baseline is supplied, a superseded committed target is more authoritative
        than the adapter's potentially stale last-ready observation.
        """
        if target_prefill is None and target_decode is None:
            raise ValueError(
                "a committed scale action must target at least one component"
            )
        if not self._is_disagg and target_prefill is not None:
            raise ValueError("aggregated replay cannot commit a prefill target")
        for name, target in (
            ("prefill", target_prefill),
            ("decode", target_decode),
        ):
            if target is not None and target < 0:
                raise ValueError(f"{name} target must be non-negative, got {target}")

        if target_prefill is not None:
            previous_target = self._scaling_target_prefill
            baseline = (
                from_prefill
                if from_prefill is not None
                else (
                    previous_target
                    if previous_target is not None
                    else self._last_ready_prefill
                )
            )
            self._scaling_target_prefill = target_prefill
            self._append_committed_event(
                at_s=at_s,
                component="prefill",
                baseline=baseline,
                target=target_prefill,
                reason=reason,
            )

        if target_decode is not None:
            previous_target = self._scaling_target_decode
            baseline = (
                from_decode
                if from_decode is not None
                else (
                    previous_target
                    if previous_target is not None
                    else self._last_ready_decode
                )
            )
            self._scaling_target_decode = target_decode
            self._append_committed_event(
                at_s=at_s,
                component="decode" if self._is_disagg else "agg",
                baseline=baseline,
                target=target_decode,
                reason=reason,
            )

        # A partner action may arrive without a local tick, so the cached
        # inventory can predate this commit even when its counts happen to match
        # the new target. Only _observe_runtime_inventory may settle it.

    def _append_committed_event(
        self,
        *,
        at_s: float,
        component: str,
        baseline: Optional[int],
        target: int,
        reason: Optional[str],
    ) -> None:
        if baseline is None or baseline == target:
            return
        direction = "scale_up" if target > baseline else "scale_down"
        logger.info(
            "Planner scaling %s: %d -> %d at t=%.1fs (%s)",
            component,
            baseline,
            target,
            at_s,
            reason or direction,
        )
        self._scaling_events.append(
            ScalingEvent(
                at_s=at_s,
                component=component,
                from_count=baseline,
                to_count=target,
                reason=reason or direction,
            )
        )

    def _is_easy_mode(self) -> bool:
        """Easy-mode check routed via config — both paths honour this
        the same way (no regression in non-SLA modes)."""
        return self._config.optimization_target != "sla"

    def _observe_runtime_inventory(self, result: dict[str, Any]) -> None:
        """Remember ready/total lifecycle counts and retire settled targets."""
        ready_p = int(result["active_prefill_count"])
        ready_d = int(result["active_decode_count"])
        self._last_ready_prefill = ready_p
        self._last_ready_decode = ready_d
        self._has_total_prefill = "total_prefill_count" in result
        self._has_total_decode = "total_decode_count" in result
        self._last_total_prefill = int(result.get("total_prefill_count", ready_p))
        self._last_total_decode = int(result.get("total_decode_count", ready_d))
        self._clear_settled_scaling_targets()

    def _clear_settled_scaling_targets(self) -> None:
        """Clear accepted targets only after ready and provisioned state settle."""

        def settled(
            target: Optional[int],
            ready: Optional[int],
            total: Optional[int],
            has_total: bool,
        ) -> bool:
            if target is None or ready is None or ready != target:
                return False
            return not has_total or total == target

        if settled(
            getattr(self, "_scaling_target_prefill", None),
            getattr(self, "_last_ready_prefill", None),
            getattr(self, "_last_total_prefill", None),
            getattr(self, "_has_total_prefill", False),
        ):
            self._scaling_target_prefill = None
        if settled(
            getattr(self, "_scaling_target_decode", None),
            getattr(self, "_last_ready_decode", None),
            getattr(self, "_last_total_decode", None),
            getattr(self, "_has_total_decode", False),
        ):
            self._scaling_target_decode = None

    def _build_tick_input(
        self, tick: ScheduledTick, result: dict[str, Any]
    ) -> TickInput:
        """Convert bridge result dict to planner TickInput."""
        # Keep planner cadence on the scheduled replay clock. The Rust bridge
        # also advances idle gaps to this timestamp so traffic windows drain
        # with the same duration the planner sees.
        now_s = tick.at_s
        self._observe_runtime_inventory(result)

        worker_counts = None
        if tick.need_worker_states:
            active_p = self._last_ready_prefill
            active_d = self._last_ready_decode
            total_p = self._last_total_prefill
            total_d = self._last_total_decode
            assert active_p is not None and active_d is not None
            assert total_p is not None and total_d is not None
            expected_p = (
                self._scaling_target_prefill
                if self._scaling_target_prefill is not None
                else total_p
            )
            expected_d = (
                self._scaling_target_decode
                if self._scaling_target_decode is not None
                else total_d
            )
            worker_counts = WorkerCounts(
                ready_num_prefill=active_p if self._is_disagg else None,
                ready_num_decode=active_d,
                expected_num_prefill=expected_p if self._is_disagg else None,
                expected_num_decode=expected_d,
                prefill_scaling_in_progress=(
                    self._is_disagg
                    and (
                        (
                            self._scaling_target_prefill is not None
                            and (
                                self._scaling_target_prefill != active_p
                                or (
                                    self._has_total_prefill
                                    and self._scaling_target_prefill != total_p
                                )
                            )
                        )
                        or (
                            self._scaling_target_prefill is None
                            and self._has_total_prefill
                            and total_p != active_p
                        )
                    )
                ),
                decode_scaling_in_progress=(
                    (
                        self._scaling_target_decode is not None
                        and (
                            self._scaling_target_decode != active_d
                            or (
                                self._has_total_decode
                                and self._scaling_target_decode != total_d
                            )
                        )
                    )
                    or (
                        self._scaling_target_decode is None
                        and self._has_total_decode
                        and total_d != active_d
                    )
                ),
            )

        fpm_observations = None
        # Merge each callback's latest worker/rank snapshots into the last-seen
        # cache, then expose the cache only on FPM ticks. This matches the live
        # subscriber's latest-snapshot semantics.
        _update_fpm_cache(
            self._prefill_fpm_cache,
            result.get("prefill_fpm_snapshots", []),
            result["active_prefill_ids"],
        )
        _update_fpm_cache(
            self._decode_fpm_cache,
            result.get("decode_fpm_snapshots", []),
            result["active_decode_ids"],
        )
        if tick.need_worker_fpm:
            prefill_dict = (
                dict(self._prefill_fpm_cache) if self._prefill_fpm_cache else None
            )
            decode_dict = (
                dict(self._decode_fpm_cache) if self._decode_fpm_cache else None
            )
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
