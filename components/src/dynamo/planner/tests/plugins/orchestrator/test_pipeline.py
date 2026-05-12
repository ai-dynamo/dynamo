# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""4-stage pipeline tests.

Covers:
- PREDICT chain_augment threads predictions into PROPOSE ctx
- PROPOSE / RECONCILE / CONSTRAIN merge happy path + baseline threading
- REJECT short-circuits the stage + rest of pipeline
- M-4: CONSTRAIN empty targets → skip_no_targets + audit event
- final priority in PROPOSE
- HOLD_LAST cache inherits to next tick
- chain-augment misuse warnings surface in audit events
- **M-7 grep-based regression test**: ``pipeline.py`` must NOT wrap
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
    CircuitState,
    ComponentTarget,
    HoldPolicy,
    OverrideResult,
    OverrideType,
    PipelineContext,
    PredictionData,
    PredictStageResponse,
    ProposeStageResponse,
    ReconcileStageResponse,
    ConstrainStageResponse,
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


def _propose_override(replicas, sub_component_type="prefill", type_=OverrideType.SET, final=False):
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
        preds = (
            None if num_req is None
            else PredictionData(predicted_num_req=num_req)
        )
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
    outcome = await orchestrator.tick(
        PipelineContext(), {PREFILL: 3}
    )
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
# M-4: empty targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_accept_on_empty_baseline_is_skip_no_targets(ctx_factory):
    # All PROPOSE plugins ACCEPT + empty baseline → CONSTRAIN produces
    # a proposal with no targets → M-4 skip_no_targets.
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
    # execution_interval=10s, HOLD_LAST → first tick runs, mid-interval tick inherits.
    orchestrator.register_internal(
        plugin_id="propose",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_propose_override(7)),
        execution_interval_seconds=10.0,
        hold_policy=HoldPolicy.HOLD_LAST,
    )
    # First tick → triggered.
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
    # mid-priority (priority=100) final=True; lower-priority-number
    # (priority=5) plugin never runs → misuse warning.
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
    assert any("chain_augment_final_misuse" in ev for ev in outcome.audit_events)


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
# M-7 grep-based regression: no stage-level asyncio.wait_for
# ---------------------------------------------------------------------------


def test_M7_pipeline_py_has_no_stage_level_wait_for():
    """Assert the pipeline source contains exactly one
    ``asyncio.wait_for`` call, and that call wraps the whole-tick body
    (``_body()``), not an ``asyncio.gather`` (M-7 invariant)."""
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
        f"Stage-level wait_for wrapping asyncio.gather is banned by M-7."
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
        first_func.attr if isinstance(first_func, ast.Attribute)
        else getattr(first_func, "id", None)
    )
    assert first_func_name != "gather", (
        "M-7 violation: asyncio.wait_for wraps asyncio.gather in pipeline.py"
    )
