# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""4-stage pipeline tests.

Covers:
- PREDICT chain_augment threads predictions into PROPOSE ctx
- PROPOSE / RECONCILE / CONSTRAIN merge happy path + baseline threading
- REJECT short-circuits the stage + rest of pipeline
- CONSTRAIN empty targets → skip_no_targets + audit event
- final priority in PROPOSE
- HOLD_LAST cache inherits to next tick
- chain-augment chain-break warnings surface in audit events
- **Grep-based regression test**: ``pipeline.py`` must NOT wrap
  ``asyncio.gather`` in ``asyncio.wait_for`` (stage-level timeouts are
  forbidden; per-plugin timeouts live in the transport).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    ConstrainStageResponse,
    HoldPolicy,
    OverrideResult,
    OverrideType,
    PipelineContext,
    PredictionData,
    PredictStageResponse,
    ProposeStageResponse,
    ReconcileStageResponse,
    RejectResult,
)

from .conftest import StubPlugin

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _propose_override(
    replicas, sub_component_type="prefill", type_=OverrideType.SET, final=False
):
    def handler(req):
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type=sub_component_type,
                        replicas=replicas,
                        type=type_,
                    )
                ]
            ),
            final=final,
        )

    return handler


def _reconcile_override(replicas, sub_component_type="prefill", type_=OverrideType.SET):
    def handler(req):
        return ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type=sub_component_type,
                        replicas=replicas,
                        type=type_,
                    )
                ]
            ),
        )

    return handler


def _constrain_at_most(replicas, sub_component_type="prefill"):
    def handler(req):
        return ConstrainStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type=sub_component_type,
                        replicas=replicas,
                        type=OverrideType.AT_MOST,
                    )
                ]
            ),
        )

    return handler


def _predict_response(num_req=None, final=False):
    def handler(req):
        preds = None if num_req is None else PredictionData(predicted_num_req=num_req)
        return PredictStageResponse(predictions=preds, final=final)

    return handler


def _accept_propose(req):
    return ProposeStageResponse(result_kind="accept", accept=AcceptResult())


def _reject_propose(reason="safety"):
    def handler(req):
        return ProposeStageResponse(
            result_kind="reject", reject=RejectResult(reason=reason)
        )

    return handler


PREFILL = ComponentKey(sub_component_type="prefill")


# ---------------------------------------------------------------------------
# Happy-path multi-stage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_output_flows_as_reconcile_baseline(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    # PROPOSE sets prefill to 7 via SET.
    orchestrator.register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(7)),
    )
    # RECONCILE has no plugins → passes PROPOSE output through unchanged.
    outcome = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 7


@pytest.mark.asyncio
async def test_constrain_at_most_clamps_propose_output(ctx_factory):
    ctx = ctx_factory()
    ctx["orchestrator"].register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(12)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="budget",
        plugin_type="constrain",
        priority=1,
        instance=StubPlugin(constrain=_constrain_at_most(8)),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 8


@pytest.mark.asyncio
async def test_broken_predict_plugin_isolated_does_not_fail_whole_tick(ctx_factory):
    """A PREDICT plugin whose ``Predict`` call raises must NOT propagate
    the exception out of the pipeline — that would kill the whole
    planner tick (regression for tedzhouhk review comment).  Instead
    ``_PredictAdapter`` records a circuit-breaker failure, emits the
    error metric, and returns a no-op response so ``chain_augment``
    moves on to the next plugin in the chain.

    Mirror of the same isolation behaviour
    ``_run_fanout_stage`` already provides for PROPOSE / RECONCILE /
    CONSTRAIN.
    """
    ctx = ctx_factory()

    def boom(_req):
        raise RuntimeError("simulated predict failure")

    # Lower priority (numerically smaller) — runs first.
    ctx["orchestrator"].register_internal(
        plugin_id="predict_broken",
        plugin_type="predict",
        priority=1,
        instance=StubPlugin(predict=boom),
    )
    # Healthy predict plugin runs after the broken one and produces
    # predictions; if isolation works the chain reaches this plugin.
    ctx["orchestrator"].register_internal(
        plugin_id="predict_healthy",
        plugin_type="predict",
        priority=2,
        instance=StubPlugin(predict=_predict_response(num_req=42.0)),
    )

    def propose_echo(req):
        # Only fires when predictions made it through the chain.
        if req.context.predictions is None:
            return ProposeStageResponse(result_kind="accept", accept=AcceptResult())
        predicted = req.context.predictions.predicted_num_req
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=int(predicted),
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    ctx["orchestrator"].register_internal(
        plugin_id="propose_echo",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=propose_echo),
    )

    # Tick must complete without raising; healthy plugin produces the
    # expected prediction; CB recorded a failure for the broken plugin.
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 42
    cb = ctx["circuit_breaker"]
    # The CB recorded a failure on the broken plugin's per-plugin
    # entry; healthy plugin's CB entry remains at zero failures.
    broken_entry = cb._entries.get("predict_broken")
    healthy_entry = cb._entries.get("predict_healthy")
    assert broken_entry is not None and broken_entry.consecutive_failures >= 1
    assert healthy_entry is not None and healthy_entry.consecutive_failures == 0


@pytest.mark.asyncio
async def test_predict_chain_threads_predictions_into_propose_context(ctx_factory):
    # PREDICT plugin sets predictions; a PROPOSE plugin that echoes the
    # running prediction into an OverrideResult demonstrates the thread.
    ctx = ctx_factory()

    def propose_from_predictions(req):
        predicted = req.context.predictions.predicted_num_req
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=int(predicted),
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    ctx["orchestrator"].register_internal(
        plugin_id="predict_one",
        plugin_type="predict",
        priority=10,
        instance=StubPlugin(predict=_predict_response(num_req=42.0)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="propose_echo",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=propose_from_predictions),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.final_proposal.targets[0].replicas == 42


# ---------------------------------------------------------------------------
# REJECT short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_reject_short_circuits(ctx_factory):
    ctx = ctx_factory()
    ctx["orchestrator"].register_internal(
        plugin_id="rej",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_reject_propose("over-capacity")),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "skip_short_circuit"
    assert outcome.final_proposal is None
    assert "over-capacity" in outcome.short_circuit_reason


# ---------------------------------------------------------------------------
# Empty-targets path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_accept_on_empty_baseline_is_skip_no_targets(ctx_factory):
    # All PROPOSE plugins ACCEPT + empty baseline → CONSTRAIN produces
    # a proposal with no targets → skip_no_targets.
    ctx = ctx_factory()
    ctx["orchestrator"].register_internal(
        plugin_id="p1",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_accept_propose),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {})
    assert outcome.execute_action == "skip_no_targets"
    assert "execute_skipped_no_targets" in outcome.audit_events


# ---------------------------------------------------------------------------
# final priority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_priority_wins_in_propose(ctx_factory):
    ctx = ctx_factory()
    # p_final (priority=5, final=True) vs p_other (priority=10, SET 99)
    ctx["orchestrator"].register_internal(
        plugin_id="p_final",
        plugin_type="propose",
        priority=5,
        instance=StubPlugin(propose=_propose_override(7, final=True)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="p_other",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(99)),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal.targets[0].replicas == 7
    assert outcome.propose_outcome.used_final_from == "p_final"


# ---------------------------------------------------------------------------
# HOLD_LAST cache inherits to next tick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_last_cache_inherits_on_idle_tick(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    clock = ctx["clock"]
    # execution_interval=10s, HOLD_LAST → first tick fires after interval
    # elapses (PSM-parity anchor), mid-interval tick inherits cache.
    orchestrator.register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(7)),
        execution_interval_seconds=10.0,
        hold_policy=HoldPolicy.HOLD_LAST,
    )
    # Advance to the first-fire moment (interval seconds since
    # registration — see test_first_fire_anchored_on_registration_time).
    clock.advance(10.0)
    first = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    assert first.final_proposal.targets[0].replicas == 7
    # Advance 5s: not due; HOLD_LAST inherits cached (7).
    clock.advance(5.0)
    second = await orchestrator.tick(PipelineContext(), {PREFILL: 3})
    assert second.execute_action == "apply"
    assert second.final_proposal.targets[0].replicas == 7


# ---------------------------------------------------------------------------
# CONSTRAIN SET dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constrain_set_dropped_and_audited(ctx_factory):
    ctx = ctx_factory()

    def constrain_set(req):
        return ConstrainStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=5,
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    ctx["orchestrator"].register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(7)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="bad_constrain",
        plugin_type="constrain",
        priority=10,
        instance=StubPlugin(constrain=constrain_set),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.constrain_outcome.set_dropped == [PREFILL]
    # SET dropped → prefill passes through as RECONCILE baseline (7).
    assert outcome.final_proposal.targets[0].replicas == 7


# ---------------------------------------------------------------------------
# chain_augment misuse warning surfaces in audit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_final_misuse_warning_surfaces_in_audit(ctx_factory):
    ctx = ctx_factory()
    # Ascending sort: emergency (priority=5) runs first as the authoritative
    # plugin (sets num_req=9.0). Then mid (priority=100) runs and sets
    # final=True → break. The misuse warning fires because mid is not the
    # lowest-priority plugin in the chain; see chain_augment module docstring
    # for why final=true at non-lowest-priority is a configuration smell.
    ctx["orchestrator"].register_internal(
        plugin_id="mid",
        plugin_type="predict",
        priority=100,
        instance=StubPlugin(predict=_predict_response(num_req=1.0, final=True)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="emergency",
        plugin_type="predict",
        priority=5,
        instance=StubPlugin(predict=_predict_response(num_req=9.0)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(3)),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    # mid's misuse message appears in audit_events.
    assert any("chain_augment_non_lowest_final" in ev for ev in outcome.audit_events)


# ---------------------------------------------------------------------------
# Tick timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whole_tick_timeout_returns_skip_tick_timeout(ctx_factory):
    import asyncio

    ctx = ctx_factory(tick_max_duration_seconds=0.05)

    async def slow_propose(req):
        await asyncio.sleep(0.5)  # exceeds tick_max
        return ProposeStageResponse(result_kind="accept", accept=AcceptResult())

    ctx["orchestrator"].register_internal(
        plugin_id="slow",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=slow_propose),
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "skip_tick_timeout"
    assert "tick_timeout_total" in outcome.audit_events
    assert outcome.final_proposal is None


# ---------------------------------------------------------------------------
# CONSTRAIN.final is ignored (proto contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constrain_final_ignored_does_not_short_circuit_merge(ctx_factory):
    """``ConstrainStageResponse.final`` is ignored per proto contract —
    constrain is a safety layer, so a constrain plugin setting
    ``final=true`` must NOT short-circuit other constrain plugins' clamps.

    Two constrain plugins:
      * A: priority=1, final=True, AT_MOST(prefill=3)
      * B: priority=5, AT_MOST(prefill=2)

    Baseline arrives at CONSTRAIN as prefill=5. If A's final were honoured,
    the merge would short-circuit at A and apply only A's clamp → prefill=3.
    Spec says both clamps must apply, monotonic → prefill=min(3, 2)=2.
    """
    ctx = ctx_factory()

    def at_most(replicas, *, final=False):
        def handler(req):
            return ConstrainStageResponse(
                result_kind="override",
                override=OverrideResult(
                    targets=[
                        ComponentTarget(
                            sub_component_type="prefill",
                            replicas=replicas,
                            type=OverrideType.AT_MOST,
                        )
                    ]
                ),
                final=final,
            )

        return handler

    # propose pushes prefill above both clamps so the AT_MOST chain visibly bites.
    ctx["orchestrator"].register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(5)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="constrain_a",
        plugin_type="constrain",
        priority=1,  # higher precedence
        instance=StubPlugin(constrain=at_most(3, final=True)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="constrain_b",
        plugin_type="constrain",
        priority=5,
        instance=StubPlugin(constrain=at_most(2)),
    )

    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 5})
    assert outcome.execute_action == "apply"
    # Both clamps applied → prefill clamped to min(3, 2) = 2.
    targets = {t.sub_component_type: t.replicas for t in outcome.final_proposal.targets}
    assert targets["prefill"] == 2


# ---------------------------------------------------------------------------
# RECONCILE proposals threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_receives_propose_results_in_proposals(ctx_factory):
    """RECONCILE plugins receive per-plugin PROPOSE results via
    ``ReconcileStageRequest.proposals``, so they can arbitrate per
    proposal rather than only seeing the post-merge ``ctx.proposal``.

    Two PROPOSE plugins emit different overrides; the RECONCILE plugin
    captures ``req.proposals`` for inspection and arbitrates: it picks
    plugin B's prefill replicas (5) even though A had higher precedence
    (priority=1) and its proposal would have won the standard merge.
    """
    captured: dict = {}

    def recording_reconcile(req):
        # Snapshot per-proposal data; assert later.
        captured["proposals"] = [
            (
                p.plugin_id,
                p.priority,
                p.result_kind,
                p.override.targets[0].replicas if p.override else None,
            )
            for p in req.proposals
        ]
        # Arbitrate: pick B's prefill (5), ignoring A's (4).
        return ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=5,
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    ctx = ctx_factory()
    ctx["orchestrator"].register_internal(
        plugin_id="propose_a",
        plugin_type="propose",
        priority=1,
        instance=StubPlugin(propose=_propose_override(4)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="propose_b",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(8)),
    )
    ctx["orchestrator"].register_internal(
        plugin_id="rec",
        plugin_type="reconcile",
        priority=10,
        instance=StubPlugin(reconcile=recording_reconcile),
    )

    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert outcome.execute_action == "apply"

    # reconcile saw BOTH propose plugins' raw results
    assert "proposals" in captured
    assert len(captured["proposals"]) == 2
    plugin_ids = {p[0] for p in captured["proposals"]}
    assert plugin_ids == {"propose_a", "propose_b"}
    # Per-plugin details preserved (priority + override replicas)
    by_id = {p[0]: p for p in captured["proposals"]}
    assert by_id["propose_a"][1] == 1  # priority
    assert by_id["propose_a"][2] == "override"
    assert by_id["propose_a"][3] == 4  # A wanted 4
    assert by_id["propose_b"][1] == 10
    assert by_id["propose_b"][3] == 8  # B wanted 8

    # RECONCILE's override took precedence — final prefill = 5 (not A's 4, not B's 8).
    targets = {t.sub_component_type: t.replicas for t in outcome.final_proposal.targets}
    assert targets["prefill"] == 5


# ---------------------------------------------------------------------------
# Grep-based regression: no stage-level asyncio.wait_for
# ---------------------------------------------------------------------------


def test_pipeline_py_has_no_stage_level_wait_for():
    """Assert the pipeline source contains exactly one
    ``asyncio.wait_for`` call, and that call wraps the whole-tick body
    (``_body()``), not an ``asyncio.gather``. Per-plugin timeouts already
    live in ``PluginTransport.call``; a stage-level wait_for would double-
    count the budget."""
    from dynamo.planner.plugins.orchestrator import pipeline as _pipeline_module

    source_path = pathlib.Path(_pipeline_module.__file__)
    assert source_path.exists(), f"pipeline source not found at {source_path}"
    tree = ast.parse(source_path.read_text())

    wait_for_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # asyncio.wait_for(...) — either Attribute or Name after "from asyncio import wait_for".
            if isinstance(func, ast.Attribute) and func.attr == "wait_for":
                wait_for_calls.append(node)
            elif isinstance(func, ast.Name) and func.id == "wait_for":
                wait_for_calls.append(node)

    assert len(wait_for_calls) == 1, (
        f"expected exactly one asyncio.wait_for in pipeline.py (the "
        f"outermost whole-tick guard); found {len(wait_for_calls)}. "
        f"Stage-level wait_for wrapping asyncio.gather is banned — "
        f"per-plugin timeouts already live in PluginTransport.call."
    )
    # The single wait_for must take a coroutine call as its first arg
    # (our outer guard calls `_body()`), not an asyncio.gather(...) result.
    call = wait_for_calls[0]
    first_arg = call.args[0]
    # Must be a Call node (calling _body()), and NOT asyncio.gather(...).
    assert isinstance(first_arg, ast.Call), (
        "the single wait_for in pipeline.py should wrap a function call "
        "(the whole-tick body), not a raw expression"
    )
    first_func = first_arg.func
    first_func_name = (
        first_func.attr
        if isinstance(first_func, ast.Attribute)
        else getattr(first_func, "id", None)
    )
    assert first_func_name != "gather", (
        "asyncio.wait_for wraps asyncio.gather in pipeline.py — "
        "stage-level deadlines are banned"
    )


# ---------------------------------------------------------------------------
# PREDICT throttle (formerly Major 5): PREDICT goes through chain_augment
# via _PredictAdapter, which is a separate dispatch path from
# _run_fanout_stage. Without the adapter calling record_evaluation,
# PREDICT plugins' last_call_at stays at -inf forever and
# execution_interval_seconds is a no-op for the entire stage.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_plugin_throttled_by_execution_interval(ctx_factory):
    """PREDICT plugin configured with ``execution_interval_seconds=60.0``
    must be skipped on subsequent ticks until the interval elapses, just
    like PROPOSE/RECONCILE/CONSTRAIN plugins.

    Pre-fix: PREDICT was never throttled because chain_augment doesn't
    touch the scheduler — ``last_call_at`` stayed at ``-math.inf``,
    ``_is_due`` always returned True, plugin fired every tick regardless
    of the configured interval.
    """
    ctx = ctx_factory()
    stub = StubPlugin(predict=_predict_response(num_req=1.0))
    ctx["registry"].register_internal(
        plugin_id="pred",
        plugin_type="predict",
        priority=1,
        instance=stub,
        execution_interval_seconds=60.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )
    # First fire happens when interval elapses since registration
    # (PSM-parity anchor — see test_first_fire_anchored_on_registration
    # _time).  Pre-PR-1 fix: first-ever fired on tick 1 regardless.
    ctx["clock"].advance(60.0)
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert stub.call_counts["Predict"] == 1
    # Second tick 1s later: must be throttled (interval is 60s).
    ctx["clock"].advance(1.0)
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert stub.call_counts["Predict"] == 1  # ← pre-throttle-fix this was 2
    # After 60s: due again.
    ctx["clock"].advance(60.0)
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert stub.call_counts["Predict"] == 2
