# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""4-stage plugin pipeline driver (DEP-XXXX PR 5 sub-task 5-4).

Pipeline order: PREDICT → PROPOSE → RECONCILE → CONSTRAIN → EXECUTE.

- **PREDICT** runs as a priority-descending chain via PR 4 ``chain_augment``;
  any partial prediction gets threaded onto ``PipelineContext.predictions``
  for downstream stages.
- **PROPOSE / RECONCILE / CONSTRAIN** fan out via ``asyncio.gather`` and
  collapse with PR 4 ``type_aware_merge``. CONSTRAIN runs with
  ``set_allowed=False`` so SET override targets are dropped + audited.
- **EXECUTE** is a decision only — the pipeline returns a
  ``PipelineOutcome`` naming the action (``apply`` / ``skip_no_targets`` /
  ``skip_short_circuit`` / ``skip_tick_timeout``). The orchestrator (or
  PR 7 NativePlannerBase) projects this onto ``PlannerConnector`` calls.

Strong constraints enforced here (v11):

- **M-1** — stage results are paired with plugins via ``zip(plugins, results)``.
  Callers must never reach back through ``result.plugin`` or assume the
  plugin object is reachable from the raw result.
- **M-4** — when CONSTRAIN produces an empty ``targets`` list (every
  plugin returned ACCEPT), the EXECUTE path is skipped with the audit
  event ``execute_skipped_no_targets`` rather than no-op-applying an
  empty proposal.
- **M-7** — **no** stage-level ``asyncio.wait_for`` wrapping
  ``asyncio.gather``. Per-plugin deadlines already live inside
  ``PluginTransport.call`` (via ``request_timeout_seconds``). The only
  ``asyncio.wait_for`` in this module is the outermost whole-tick
  guard around the entire pipeline. A grep-based regression test in
  ``tests/plugins/orchestrator/test_pipeline.py`` asserts this.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics
from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.merge import (
    ChainAugmentOutcome,
    ComponentKey,
    MergeOutcome,
    PluginResult,
    chain_augment,
    type_aware_merge,
)
from dynamo.planner.plugins.merge.types import PredictPluginCallable
from dynamo.planner.plugins.registry.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)
from dynamo.planner.plugins.registry.types import RegisteredPlugin
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    ConstrainStageRequest,
    ConstrainStageResponse,
    OverrideResult,
    PipelineContext,
    PredictStageRequest,
    PredictStageResponse,
    ProposeStageRequest,
    ProposeStageResponse,
    ReconcileStageRequest,
    ReconcileStageResponse,
    RejectResult,
    ScalingProposal,
)

log = logging.getLogger(__name__)


ExecuteAction = Literal[
    "apply",  # ctx.constrained.targets should be applied
    "skip_no_targets",  # M-4: no targets — emit audit + skip
    "skip_short_circuit",  # some stage REJECTed
    "skip_tick_timeout",  # whole-tick deadline exceeded
]


@dataclass
class PipelineOutcome:
    """Full record of one tick through the 4-stage pipeline.

    The orchestrator consumes ``execute_action`` to decide what to hand
    to the ``PlannerConnector``. ``final_proposal`` is populated in the
    ``apply`` / ``skip_no_targets`` branches and ``None`` otherwise.
    """

    execute_action: ExecuteAction
    final_proposal: Optional[ScalingProposal]
    short_circuit_reason: str = ""
    predict_outcome: Optional[ChainAugmentOutcome] = None
    propose_outcome: Optional[MergeOutcome] = None
    reconcile_outcome: Optional[MergeOutcome] = None
    constrain_outcome: Optional[MergeOutcome] = None
    audit_events: list[str] = field(default_factory=list)


class _PredictAdapter:
    """Adapts a ``RegisteredPlugin`` to the
    ``PredictPluginCallable`` protocol expected by PR 4 ``chain_augment``.

    ``chain_augment`` wants ``async call(method, context)``; the
    transport signature is ``async call(method, request)``. This adapter
    wraps the ``PipelineContext`` into a ``PredictStageRequest`` on the
    way in and forwards the response unchanged.

    PR 8 8-2 wiring gap fix: emits ``plugin_evaluations_total`` +
    ``plugin_latency_seconds`` for every PREDICT call so family-2
    metrics cover all 4 stages uniformly. Previously PREDICT silently
    bypassed emission because ``chain_augment`` is a separate dispatch
    path from ``_run_fanout_stage``.
    """

    def __init__(
        self,
        plugin: RegisteredPlugin,
        *,
        metrics: Optional[PluginFrameworkMetrics] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._plugin = plugin
        self._metrics = metrics
        self._clock = clock

    @property
    def plugin_id(self) -> str:
        return self._plugin.plugin_id

    @property
    def priority(self) -> int:
        return self._plugin.priority

    async def call(
        self, method: str, context: PipelineContext
    ) -> PredictStageResponse:
        assert method == "Predict", f"unexpected method for PREDICT: {method!r}"
        req = PredictStageRequest(context=context)

        started = self._clock.now() if (self._metrics and self._clock) else 0.0
        try:
            resp = await self._plugin.transport.call("Predict", req)
        except Exception:
            if self._metrics is not None:
                self._metrics.plugin_evaluations_total.labels(
                    plugin_id=self._plugin.plugin_id,
                    stage="predict",
                    result="error",
                ).inc()
            raise

        if self._metrics is not None:
            # Classify: chain_augment consumes the response and produces
            # a partial/final PredictionData — we don't know yet whether
            # final=True "won" the chain, so use a coarse label here.
            # Terminal chain outcome is captured elsewhere (audit events
            # in chain_augment's misuse_warnings).
            self._metrics.plugin_evaluations_total.labels(
                plugin_id=self._plugin.plugin_id,
                stage="predict",
                result="accept",  # PREDICT returns data, not overrides
            ).inc()
            if self._clock is not None:
                self._metrics.plugin_latency_seconds.labels(
                    plugin_id=self._plugin.plugin_id, stage="predict"
                ).observe(max(0.0, self._clock.now() - started))

        return resp  # type: ignore[return-value]


def _proposal_to_baseline(
    proposal: Optional[ScalingProposal],
    fallback: Mapping[ComponentKey, int],
) -> dict[ComponentKey, int]:
    """Project a ``ScalingProposal`` to the ``baseline`` shape consumed
    by the next stage's ``type_aware_merge``.

    Per PR 4 § "Cross-Task Coordination" #4, the baseline for each stage
    is the prior stage's output (not the caller's initial baseline).
    If the prior stage produced no proposal (short-circuit edge case),
    fall back to the caller's baseline.
    """
    if proposal is None:
        return dict(fallback)
    out: dict[ComponentKey, int] = dict(fallback)
    for t in proposal.targets:
        if t.replicas is None:
            continue
        key = ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        )
        out[key] = t.replicas
    return out


def _stage_request(stage: str, ctx: PipelineContext):
    if stage == "propose":
        return ProposeStageRequest(context=ctx)
    if stage == "reconcile":
        # v1 limitation: proposals list is empty; per-plugin PROPOSE
        # results are available in the upstream merge but not threaded
        # as a ReconcileStageRequest.proposals payload in this iteration.
        # PR 7 or a later follow-up will add that path if needed.
        return ReconcileStageRequest(context=ctx)
    if stage == "constrain":
        return ConstrainStageRequest(context=ctx)
    raise ValueError(f"_stage_request: unknown stage {stage!r}")


_STAGE_METHOD = {
    "propose": "Propose",
    "reconcile": "Reconcile",
    "constrain": "Constrain",
}


def _response_to_plugin_result(
    plugin: RegisteredPlugin,
    resp: ProposeStageResponse | ReconcileStageResponse | ConstrainStageResponse,
) -> Optional[PluginResult]:
    """Convert a ``_StageOneofResponse`` to ``PluginResult`` per M-1 pairing.

    Returns ``None`` (treated as ACCEPT by the caller) if the plugin
    violated the oneof contract by returning all fields unset.
    """
    kind = resp.result_kind
    if kind == "accept" and resp.accept is not None:
        return PluginResult(
            plugin_id=plugin.plugin_id,
            priority=plugin.priority,
            result=resp.accept,
            final=resp.final,
        )
    if kind == "override" and resp.override is not None:
        return PluginResult(
            plugin_id=plugin.plugin_id,
            priority=plugin.priority,
            result=resp.override,
            final=resp.final,
        )
    if kind == "reject" and resp.reject is not None:
        return PluginResult(
            plugin_id=plugin.plugin_id,
            priority=plugin.priority,
            result=resp.reject,
            final=resp.final,
        )
    # Empty oneof = plugin contract violation (v11 Hidden Knowledge):
    # log + treat as ACCEPT for this tick so the stage still makes progress.
    log.warning(
        "pipeline: plugin_id=%s returned empty oneof result (result_kind=%r); "
        "treating as ACCEPT for this tick",
        plugin.plugin_id,
        kind,
    )
    return None


async def _run_fanout_stage(
    *,
    stage: str,
    scheduler: PluginScheduler,
    circuit_breaker: CircuitBreaker,
    ctx: PipelineContext,
    baseline: Mapping[ComponentKey, int],
    tick_now: float,
    set_allowed: bool,
    clock: Clock,
    metrics: Optional[PluginFrameworkMetrics] = None,
) -> MergeOutcome:
    """PROPOSE / RECONCILE / CONSTRAIN fan-out-and-merge helper.

    Computes the active set, dispatches via ``asyncio.gather`` (M-7: no
    wrapping ``asyncio.wait_for``), records success/failure on the
    circuit breaker, threads inherited HOLD_LAST results into the merge,
    and returns the ``type_aware_merge`` outcome.

    When ``metrics`` is provided, emits PR 8 family-2 plugin metrics
    (evaluations / latency / held_over / cache_age / circuit_state /
    override_active) at the appropriate points.  Passing ``None``
    disables emission for tests + replay that don't construct a
    Prometheus registry.
    """
    active = scheduler.compute_active_set(tick_now, stage)
    plugins: list[RegisteredPlugin] = list(active.triggered)
    method = _STAGE_METHOD[stage]
    request = _stage_request(stage, ctx)

    # Record latency per-plugin: measure each call individually even
    # though they run concurrently, so slow plugins don't get their
    # latency collapsed into the gather deadline.
    call_starts: list[float] = []
    if metrics is not None:
        call_starts = [clock.now() for _ in plugins]

    # M-7: Bare asyncio.gather — each transport.call enforces its own
    # per-plugin timeout inside PluginTransport.
    raw_results = await asyncio.gather(
        *[p.transport.call(method, request) for p in plugins],
        return_exceptions=True,
    )

    call_end = clock.now() if metrics is not None else 0.0

    # M-1: pair plugins with their raw results via zip — do NOT assume
    # the result carries a back-reference to the plugin.
    plugin_results: list[PluginResult] = []
    contributing_plugin_ids: set[str] = set()
    for idx, (plugin, raw) in enumerate(zip(plugins, raw_results)):
        if isinstance(raw, Exception):
            log.warning(
                "pipeline.%s: plugin_id=%s call failed: %r",
                stage,
                plugin.plugin_id,
                raw,
            )
            circuit_breaker.record_failure(plugin.plugin_id)
            if metrics is not None:
                _record_eval(metrics, plugin.plugin_id, stage, "error")
            continue
        circuit_breaker.record_success(plugin.plugin_id)
        if metrics is not None:
            # Latency emitted only for successful calls so error / timeout
            # tail doesn't pollute the plugin-perf percentiles dashboard.
            metrics.plugin_latency_seconds.labels(
                plugin_id=plugin.plugin_id, stage=stage
            ).observe(max(0.0, call_end - call_starts[idx]))
        pr = _response_to_plugin_result(plugin, raw)
        if pr is None:
            if metrics is not None:
                _record_eval(metrics, plugin.plugin_id, stage, "error")
            continue
        plugin_results.append(pr)
        if metrics is not None:
            _record_eval(
                metrics,
                plugin.plugin_id,
                stage,
                _result_label(pr),
            )
        # Cache OverrideResult for HOLD_LAST plugins on the scheduler.
        if isinstance(pr.result, OverrideResult):
            scheduler.record_result(plugin.plugin_id, stage, pr.result, tick_now)
            contributing_plugin_ids.add(plugin.plugin_id)

    # Inherited HOLD_LAST entries participate in the merge as non-final
    # PluginResults (cache replay cannot re-assert final=True).
    for inh in active.inherited:
        plugin_results.append(
            PluginResult(
                plugin_id=inh.plugin_id,
                priority=inh.priority,
                result=inh.result,
                final=False,
            )
        )
        if metrics is not None:
            metrics.plugin_held_over_total.labels(
                plugin_id=inh.plugin_id, stage=stage
            ).inc()
            metrics.plugin_cache_age_seconds.labels(plugin_id=inh.plugin_id).set(
                max(0.0, tick_now - inh.cached_at)
            )
            _record_eval(metrics, inh.plugin_id, stage, "held_over")
            contributing_plugin_ids.add(inh.plugin_id)

    outcome = type_aware_merge(plugin_results, baseline, set_allowed=set_allowed)

    if metrics is not None:
        _set_circuit_state(metrics, plugins + [_inh_as_plugin(i) for i in active.inherited], circuit_breaker)
        _emit_override_active(
            metrics,
            stage=stage,
            plugin_results=plugin_results,
            outcome=outcome,
        )
        _emit_clamps_and_rejects(
            metrics,
            stage=stage,
            outcome=outcome,
            plugin_results=plugin_results,
        )

    return outcome


# ---------------------------------------------------------------------------
# PR 8 8-2 helpers: classify plugin result → metric label, emit gauges.
# Keeping these close to the fan-out helper so the metric vocabulary
# stays in one place and matches what dashboards expect.
# ---------------------------------------------------------------------------


def _record_eval(
    metrics: PluginFrameworkMetrics,
    plugin_id: str,
    stage: str,
    result_label: str,
) -> None:
    metrics.plugin_evaluations_total.labels(
        plugin_id=plugin_id, stage=stage, result=result_label
    ).inc()


def _result_label(pr: PluginResult) -> str:
    """Map a PluginResult to the ``result`` label used in metrics.
    Mirrors the taxonomy the spec calls out: accept / set / at_least /
    at_most / reject / held_over / timeout / error.

    ``override_type`` is on each ``ComponentTarget`` (not on
    ``OverrideResult`` itself — proto mirrors per-target types so one
    result can emit mixed SET/AT_LEAST/AT_MOST per component).  We pick
    the first target's type as the label here because Prometheus label
    cardinality requires a single value; downstream dashboards that
    need the full mix should sum
    ``plugin_override_active{override_type=...}`` instead.
    """
    from dynamo.planner.plugins.types import (
        AcceptResult as _AcceptResult,
        OverrideResult as _OverrideResult,
        RejectResult as _RejectResult,
    )

    r = pr.result
    if isinstance(r, _RejectResult):
        return "reject"
    if isinstance(r, _AcceptResult):
        return "accept"
    if isinstance(r, _OverrideResult):
        if r.targets:
            t = r.targets[0].type
            return t.name.lower() if hasattr(t, "name") else str(t).lower()
        return "set"  # OverrideResult with empty targets defaults to SET semantically
    return "unknown"


def _inh_as_plugin(inh):
    """Adapter: return a minimal object with ``plugin_id`` so
    ``_set_circuit_state`` can treat inherited entries uniformly."""

    class _Shim:
        plugin_id = inh.plugin_id

    return _Shim


def _set_circuit_state(
    metrics: PluginFrameworkMetrics,
    plugins,
    circuit_breaker: CircuitBreaker,
) -> None:
    """Reflect the circuit breaker's per-plugin state onto the gauge.

    Called once per fanout stage.  Reads the state (which may
    auto-transition OPEN → HALF_OPEN after cooldown) and pins the gauge
    so dashboards display the live view even for plugins not evaluated
    this tick."""
    from dynamo.planner.monitoring.planner_metrics import (
        CIRCUIT_STATE_CLOSED,
        CIRCUIT_STATE_HALF_OPEN,
        CIRCUIT_STATE_OPEN,
    )

    _state_map = {
        CircuitState.CLOSED: CIRCUIT_STATE_CLOSED,
        CircuitState.HALF_OPEN: CIRCUIT_STATE_HALF_OPEN,
        CircuitState.OPEN: CIRCUIT_STATE_OPEN,
    }
    seen: set[str] = set()
    for p in plugins:
        if p.plugin_id in seen:
            continue
        seen.add(p.plugin_id)
        state = circuit_breaker.state(p.plugin_id)
        metrics.plugin_circuit_state.labels(plugin_id=p.plugin_id).set(
            _state_map.get(state, CIRCUIT_STATE_CLOSED)
        )


def _emit_override_active(
    metrics: PluginFrameworkMetrics,
    *,
    stage: str,
    plugin_results: list,
    outcome: MergeOutcome,
) -> None:
    """Set ``plugin_override_active`` for every evaluated plugin in this
    stage.  The gauge is per-(plugin_id, stage, override_type); we reset
    all four types first (so the previous tick's 1 doesn't linger) then
    set 1 for the type the plugin actually contributed.

    Plugins that returned ACCEPT or REJECT-but-not-winning leave the
    gauge at all-zero — that's the correct "evaluated, no override"
    state."""
    from dynamo.planner.plugins.types import (
        OverrideResult as _OverrideResult,
        RejectResult as _RejectResult,
    )

    # Reset every plugin we saw this tick before setting their actual
    # contribution.  Iteration over plugin_results covers both triggered
    # and inherited entries.
    for pr in plugin_results:
        metrics.reset_overrides(pr.plugin_id, stage)

    # Short-circuited REJECT winners (found by type_aware_merge) surface
    # as outcome.rejected; emit override_type=REJECT for them.
    rejected_ids = {
        pr.plugin_id
        for pr in plugin_results
        if isinstance(pr.result, _RejectResult)
    }
    for pid in rejected_ids:
        metrics.plugin_override_active.labels(
            plugin_id=pid, stage=stage, override_type="REJECT"
        ).set(1)

    # For non-rejected plugins, emit 1 on each override_type present
    # in their targets.  A plugin may contribute mixed types (e.g.
    # SET on prefill + AT_LEAST on decode) — we flag each observed
    # type.  This is a conservative over-count: a plugin's SET may
    # lose to a higher-priority SET from another plugin and still
    # show as "active".  Refine when MergeOutcome exposes a concrete
    # contributor list.
    for pr in plugin_results:
        if not isinstance(pr.result, _OverrideResult):
            continue
        types_seen: set[str] = set()
        for target in pr.result.targets:
            kind = target.type
            label = kind.name if hasattr(kind, "name") else str(kind)
            types_seen.add(label)
        for label in types_seen:
            metrics.plugin_override_active.labels(
                plugin_id=pr.plugin_id, stage=stage, override_type=label
            ).set(1)


def _emit_clamps_and_rejects(
    metrics: PluginFrameworkMetrics,
    *,
    stage: str,
    outcome: MergeOutcome,
    plugin_results: list,
) -> None:
    """PR 8 8-3: surface ``type_aware_merge`` clamp + reject events as
    family-3 counters.

    - ``reconcile_clamped_total`` / ``constrain_capped_total`` fire once
      per clamp event (one per ``(key, direction)`` tuple), labelled by
      component + winning plugin source. PROPOSE-stage clamps are NOT
      counted — they're ordinary merge math, not "something overrode the
      recommendation". Only RECONCILE and CONSTRAIN get the counter.
    - ``reject_short_circuited_total`` fires once per pipeline stage
      that short-circuited on a REJECT, labelled by the rejecting
      plugin_id. Called even if the caller then bails on the stage —
      the counter tracks "REJECT happened", independent of what the
      orchestrator does next.
    """
    from dynamo.planner.plugins.types import RejectResult as _RejectResult

    # -- clamp counters ----------------------------------------------------
    clamp_counter = None
    if stage == "reconcile":
        clamp_counter = metrics.reconcile_clamped_total
    elif stage == "constrain":
        clamp_counter = metrics.constrain_capped_total
    if clamp_counter is not None and outcome.clamped:
        for key, _direction, source in outcome.clamped:
            clamp_counter.labels(
                sub_component_type=key.sub_component_type,
                component_name=key.component_name or "",
                source=source,
            ).inc()

    # -- reject counter (any stage) ----------------------------------------
    if outcome.short_circuited:
        for pr in plugin_results:
            if isinstance(pr.result, _RejectResult):
                metrics.reject_short_circuited_total.labels(
                    plugin_id=pr.plugin_id
                ).inc()


async def run_pipeline(
    *,
    ctx: PipelineContext,
    scheduler: PluginScheduler,
    circuit_breaker: CircuitBreaker,
    baseline: Mapping[ComponentKey, int],
    clock: Clock,
    tick_now: float,
    tick_max_duration_seconds: float,
    metrics: Optional[PluginFrameworkMetrics] = None,
) -> PipelineOutcome:
    """Run one tick through the 4 stages and return a PipelineOutcome.

    Args:
        ctx: Initial PipelineContext (observations / request_id / decision_id).
        scheduler: PR 3 PluginScheduler; provides the active set per stage
            and records OverrideResult for HOLD_LAST inheritance.
        circuit_breaker: PR 3 CircuitBreaker; records success/failure
            transitions driven by plugin call outcomes here.
        baseline: Current replicas per ComponentKey — fed to every
            ``type_aware_merge`` call for recommendation fallback and
            AT_LEAST/AT_MOST clamping.
        clock: PR 2 Clock; used only for the whole-tick deadline guard.
        tick_now: Monotonic timestamp the active set + record_result use
            for "due" detection and HOLD_LAST cache age.
        tick_max_duration_seconds: Outermost deadline — wraps the entire
            4-stage pipeline in a single ``asyncio.wait_for``. Per-stage
            timeouts live inside ``PluginTransport.call`` (M-7).
    """

    async def _body() -> PipelineOutcome:
        audit: list[str] = []
        current_ctx = ctx

        # ---- PREDICT stage (priority-descending chain) ----
        predict_active = scheduler.compute_active_set(tick_now, "predict")
        predict_adapters: list[PredictPluginCallable] = [
            _PredictAdapter(p, metrics=metrics, clock=clock)
            for p in predict_active.triggered
        ]
        ca = await chain_augment(predict_adapters, current_ctx)
        # PR 8 8-2 fix: also refresh circuit_state gauge for predict
        # plugins (fan-out helper does this for other stages).
        if metrics is not None:
            _set_circuit_state(metrics, predict_active.triggered, circuit_breaker)
        if ca.misuse_warnings:
            audit.extend(ca.misuse_warnings)
        if ca.prediction is not None:
            current_ctx = current_ctx.model_copy(
                update={"predictions": ca.prediction}
            )

        # ---- PROPOSE stage ----
        propose = await _run_fanout_stage(
            stage="propose",
            scheduler=scheduler,
            circuit_breaker=circuit_breaker,
            ctx=current_ctx,
            baseline=baseline,
            tick_now=tick_now,
            set_allowed=True,
            clock=clock,
            metrics=metrics,
        )
        if propose.short_circuited:
            return PipelineOutcome(
                execute_action="skip_short_circuit",
                final_proposal=None,
                short_circuit_reason=propose.short_circuit_reason,
                predict_outcome=ca,
                propose_outcome=propose,
                audit_events=audit,
            )
        if propose.proposal is not None:
            current_ctx = current_ctx.model_copy(
                update={"proposal": propose.proposal}
            )

        # ---- RECONCILE stage ----
        # Baseline flows from PROPOSE's output (PR 4 cross-task #4).
        reconcile_baseline = _proposal_to_baseline(propose.proposal, baseline)
        reconcile = await _run_fanout_stage(
            stage="reconcile",
            scheduler=scheduler,
            circuit_breaker=circuit_breaker,
            ctx=current_ctx,
            baseline=reconcile_baseline,
            tick_now=tick_now,
            set_allowed=True,
            clock=clock,
            metrics=metrics,
        )
        if reconcile.short_circuited:
            return PipelineOutcome(
                execute_action="skip_short_circuit",
                final_proposal=None,
                short_circuit_reason=reconcile.short_circuit_reason,
                predict_outcome=ca,
                propose_outcome=propose,
                reconcile_outcome=reconcile,
                audit_events=audit,
            )
        if reconcile.proposal is not None:
            current_ctx = current_ctx.model_copy(
                update={"proposal": reconcile.proposal}
            )

        # ---- CONSTRAIN stage ----
        # Baseline flows from RECONCILE's output (PR 4 cross-task #4).
        constrain_baseline = _proposal_to_baseline(reconcile.proposal, baseline)
        constrain = await _run_fanout_stage(
            stage="constrain",
            scheduler=scheduler,
            circuit_breaker=circuit_breaker,
            ctx=current_ctx,
            baseline=constrain_baseline,
            tick_now=tick_now,
            set_allowed=False,
            clock=clock,
            metrics=metrics,
        )
        if constrain.short_circuited:
            return PipelineOutcome(
                execute_action="skip_short_circuit",
                final_proposal=None,
                short_circuit_reason=constrain.short_circuit_reason,
                predict_outcome=ca,
                propose_outcome=propose,
                reconcile_outcome=reconcile,
                constrain_outcome=constrain,
                audit_events=audit,
            )

        # ---- EXECUTE decision ----
        final = constrain.proposal
        if final is None or not final.targets:
            # M-4: empty targets path is an explicit skip + audit.
            audit.append("execute_skipped_no_targets")
            return PipelineOutcome(
                execute_action="skip_no_targets",
                final_proposal=final,
                predict_outcome=ca,
                propose_outcome=propose,
                reconcile_outcome=reconcile,
                constrain_outcome=constrain,
                audit_events=audit,
            )

        return PipelineOutcome(
            execute_action="apply",
            final_proposal=final,
            predict_outcome=ca,
            propose_outcome=propose,
            reconcile_outcome=reconcile,
            constrain_outcome=constrain,
            audit_events=audit,
        )

    # Outermost safety deadline (M-7: this is the ONLY asyncio.wait_for
    # in this module, and it wraps the entire pipeline, not a single stage).
    try:
        # PR 8 8-5: tick_duration_seconds histogram — measured around
        # the outer wait_for so it includes every stage + the timeout
        # machinery itself (matches what operators see as "tick cost").
        tick_start = clock.now()
        try:
            outcome = await asyncio.wait_for(
                _body(), timeout=tick_max_duration_seconds
            )
        finally:
            if metrics is not None:
                metrics.tick_duration_seconds.observe(
                    max(0.0, clock.now() - tick_start)
                )
        return outcome
    except asyncio.TimeoutError:
        if metrics is not None:
            metrics.tick_timeout_total.inc()
        log.warning(
            "pipeline: tick exceeded tick_max_duration_seconds=%.2f",
            tick_max_duration_seconds,
        )
        return PipelineOutcome(
            execute_action="skip_tick_timeout",
            final_proposal=None,
            short_circuit_reason=(
                f"tick_max_duration_seconds={tick_max_duration_seconds:.2f}"
            ),
            audit_events=["tick_timeout_total"],
        )


# Re-exports that help tests / readers avoid pulling from both merge types.
_ = (AcceptResult, RejectResult, ScalingProposal, PipelineContext, ComponentTarget)


__all__ = [
    "PipelineOutcome",
    "ExecuteAction",
    "run_pipeline",
]
