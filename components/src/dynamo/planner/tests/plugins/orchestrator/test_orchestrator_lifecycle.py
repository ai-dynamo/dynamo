# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle + regression-model accessor tests for LocalPlannerOrchestrator."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.types import (
    ComponentTarget,
    HoldPolicy,
    OverrideResult,
    OverrideType,
    PipelineContext,
    ProposeStageResponse,
)

from .conftest import StubPlugin

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_propose_stub(replicas):
    def handler(req):
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        replicas=replicas,
                        type=OverrideType.SET,
                    )
                ]
            ),
        )

    return handler


# ---------------------------------------------------------------------------
# Construction + invalid config
# ---------------------------------------------------------------------------


def test_zero_tick_max_duration_rejected(ctx_factory):
    with pytest.raises(ValueError):
        ctx_factory(tick_max_duration_seconds=0)


def test_orchestrator_starts_with_empty_plugin_set(ctx_factory):
    ctx = ctx_factory()
    assert ctx["orchestrator"].list_plugins() == []


# ---------------------------------------------------------------------------
# register_internal
# ---------------------------------------------------------------------------


def test_register_internal_via_orchestrator_appears_in_list(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    orchestrator.register_internal(
        plugin_id="stub",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_make_propose_stub(5)),
    )
    infos = orchestrator.list_plugins()
    assert [i.plugin_id for i in infos] == ["stub"]
    assert infos[0].is_builtin is True
    assert infos[0].transport == "in_process"


# ---------------------------------------------------------------------------
# Regression-model accessors
# ---------------------------------------------------------------------------


def test_get_regression_returns_none_for_unknown_kind(ctx_factory):
    ctx = ctx_factory()
    assert ctx["orchestrator"].get_regression("prefill") is None


def test_update_then_get_regression_returns_same_reference(ctx_factory):
    ctx = ctx_factory()
    model = object()
    ctx["orchestrator"].update_regression("prefill", model)
    assert ctx["orchestrator"].get_regression("prefill") is model


def test_update_regression_replaces_existing(ctx_factory):
    ctx = ctx_factory()
    ctx["orchestrator"].update_regression("prefill", "v1")
    ctx["orchestrator"].update_regression("prefill", "v2")
    assert ctx["orchestrator"].get_regression("prefill") == "v2"


# ---------------------------------------------------------------------------
# tick happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_happy_path_single_propose_plugin(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    orchestrator.register_internal(
        plugin_id="propose_one",
        plugin_type="propose",
        priority=10,
        instance=StubPlugin(propose=_make_propose_stub(7)),
    )
    baseline = {ComponentKey(sub_component_type="prefill"): 3}
    outcome = await orchestrator.tick(PipelineContext(), baseline)
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal is not None
    assert outcome.final_proposal.targets[0].replicas == 7


@pytest.mark.asyncio
async def test_tick_with_no_plugins_applies_baseline(ctx_factory):
    # No plugins registered in any stage → type_aware_merge emits a
    # pass-through proposal from the baseline, so EXECUTE applies it.
    ctx = ctx_factory()
    baseline = {ComponentKey(sub_component_type="prefill"): 5}
    outcome = await ctx["orchestrator"].tick(PipelineContext(), baseline)
    assert outcome.execute_action == "apply"
    assert outcome.final_proposal is not None
    assert outcome.final_proposal.targets[0].replicas == 5


@pytest.mark.asyncio
async def test_tick_empty_baseline_and_no_plugins_is_skip_no_targets(
    ctx_factory,
):
    # Empty baseline + no plugins → CONSTRAIN proposal.targets == [] →
    # Empty-targets path: skip_no_targets + audit event.
    ctx = ctx_factory()
    outcome = await ctx["orchestrator"].tick(PipelineContext(), baseline={})
    assert outcome.execute_action == "skip_no_targets"
    assert "execute_skipped_no_targets" in outcome.audit_events


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_unregisters_all(ctx_factory):
    ctx = ctx_factory()
    orchestrator = ctx["orchestrator"]
    orchestrator.register_internal(
        plugin_id="a",
        plugin_type="propose",
        priority=1,
        instance=StubPlugin(propose=_make_propose_stub(1)),
    )
    orchestrator.register_internal(
        plugin_id="b",
        plugin_type="propose",
        priority=2,
        instance=StubPlugin(propose=_make_propose_stub(2)),
    )
    assert len(orchestrator.list_plugins()) == 2
    await orchestrator.shutdown()
    assert orchestrator.list_plugins() == []


@pytest.mark.asyncio
async def test_shutdown_is_idempotent(ctx_factory):
    ctx = ctx_factory()
    await ctx["orchestrator"].shutdown()
    await ctx["orchestrator"].shutdown()
    assert ctx["orchestrator"].list_plugins() == []
