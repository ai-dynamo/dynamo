# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BuiltinBudgetConstrain (PR 6 6-6).

The plugin emits ``AT_LEAST`` / ``AT_MOST`` targets; the test validates
both the **shape** of emitted targets and the **merge outcome** when
piped through PR 4 ``type_aware_merge`` against a stand-in upstream
proposal — i.e. does the plugin produce the correct clamp behaviour?
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.builtins.budget_constrain import BuiltinBudgetConstrain
from dynamo.planner.plugins.merge import (
    ComponentKey,
    PluginResult,
    type_aware_merge,
)
from dynamo.planner.plugins.types import (
    ComponentTarget,
    ConstrainStageRequest,
    OverrideResult,
    OverrideType,
    PipelineContext,
    ScalingProposal,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeOrchestrator:
    def __init__(self, caps):
        self._caps = caps

    @property
    def capabilities(self):
        return self._caps

    def get_regression(self, kind):
        return None

    def update_regression(self, kind, model):
        pass


def _caps_disagg(p_gpu=1, d_gpu=1):
    return WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=p_gpu, max_num_batched_tokens=2048),
        decode=EngineCapabilities(num_gpu=d_gpu, max_num_batched_tokens=2048),
    )


def _caps_agg(d_gpu=1):
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=d_gpu, max_num_batched_tokens=2048),
    )


def _config(**kw):
    return PlannerConfig(
        mode=kw.pop("mode", "disagg"),
        enable_throughput_scaling=kw.pop("enable_throughput_scaling", True),
        enable_load_scaling=kw.pop("enable_load_scaling", False),
        optimization_target=kw.pop("optimization_target", "sla"),
        max_gpu_budget=kw.pop("max_gpu_budget", -1),
        min_endpoint=kw.pop("min_endpoint", 1),
        **kw,
    )


def _merge_with_upstream(
    plugin_result, upstream_proposal=None, baseline=None
):
    """Run PR 4 type_aware_merge with the plugin's result + an optional
    upstream proposal. CONSTRAIN mode (set_allowed=False) drops SET
    plugin outputs; the real pipeline threads the upstream proposal into
    the **baseline** dict (see pipeline._proposal_to_baseline), so we
    mirror that here — upstream_proposal.targets become the baseline
    map fed to type_aware_merge."""
    results = [plugin_result]
    base = dict(baseline or {})
    if upstream_proposal is not None:
        for t in upstream_proposal.targets:
            if t.replicas is None:
                continue
            key = ComponentKey(
                sub_component_type=t.sub_component_type,
                component_name=t.component_name,
            )
            base[key] = t.replicas
    return type_aware_merge(results, base, set_allowed=False)


def _replicas_by_key(proposal):
    return {
        ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        ): t.replicas
        for t in proposal.targets
    }


# ---------------------------------------------------------------------------
# Shape: min_endpoint AT_LEAST + max_budget AT_MOST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_min_endpoint_at_least_per_component():
    config = _config(max_gpu_budget=-1, min_endpoint=2)
    plugin = BuiltinBudgetConstrain(_FakeOrchestrator(_caps_disagg()), config)  # type: ignore[arg-type]
    resp = await plugin.Constrain(ConstrainStageRequest())
    assert resp.result_kind == "override"
    # AT_LEAST=2 for both prefill and decode; no AT_MOST (budget = -1).
    kinds = [(t.sub_component_type, t.type, t.replicas) for t in resp.override.targets]
    assert ("prefill", OverrideType.AT_LEAST, 2) in kinds
    assert ("decode", OverrideType.AT_LEAST, 2) in kinds
    assert not any(t.type == OverrideType.AT_MOST for t in resp.override.targets)


@pytest.mark.asyncio
async def test_emits_at_most_when_budget_configured():
    # budget=10, p_gpu=2, d_gpu=1, min_endpoint=1
    # ceiling_p = (10 - 1*1) // 2 = 4
    # ceiling_d = (10 - 1*2) // 1 = 8
    config = _config(max_gpu_budget=10, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_disagg(p_gpu=2, d_gpu=1)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    assert resp.result_kind == "override"
    at_most = [
        (t.sub_component_type, t.replicas)
        for t in resp.override.targets
        if t.type == OverrideType.AT_MOST
    ]
    assert ("prefill", 4) in at_most
    assert ("decode", 8) in at_most


# ---------------------------------------------------------------------------
# Merge outcome: upstream clamped correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_clamps_high_upstream_to_ceiling():
    config = _config(max_gpu_budget=10, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_disagg(p_gpu=2, d_gpu=1)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    plugin_pr = PluginResult(
        plugin_id="budget", priority=1, result=resp.override, final=False
    )

    # Upstream wants prefill=10, decode=20 — should clamp to (4, 8).
    upstream = ScalingProposal(
        targets=[
            ComponentTarget(sub_component_type="prefill", replicas=10, type=OverrideType.SET),
            ComponentTarget(sub_component_type="decode", replicas=20, type=OverrideType.SET),
        ]
    )
    outcome = _merge_with_upstream(plugin_pr, upstream)
    assert outcome.short_circuited is False
    assert _replicas_by_key(outcome.proposal) == {
        ComponentKey(sub_component_type="prefill"): 4,
        ComponentKey(sub_component_type="decode"): 8,
    }


@pytest.mark.asyncio
async def test_merge_lifts_below_min_endpoint_to_floor():
    config = _config(max_gpu_budget=-1, min_endpoint=2)
    plugin = BuiltinBudgetConstrain(_FakeOrchestrator(_caps_disagg()), config)  # type: ignore[arg-type]
    resp = await plugin.Constrain(ConstrainStageRequest())
    plugin_pr = PluginResult(
        plugin_id="budget", priority=1, result=resp.override, final=False
    )

    # Upstream wants 0 replicas — floor lifts to min_endpoint=2.
    upstream = ScalingProposal(
        targets=[
            ComponentTarget(sub_component_type="prefill", replicas=0, type=OverrideType.SET),
            ComponentTarget(sub_component_type="decode", replicas=0, type=OverrideType.SET),
        ]
    )
    outcome = _merge_with_upstream(plugin_pr, upstream)
    assert _replicas_by_key(outcome.proposal) == {
        ComponentKey(sub_component_type="prefill"): 2,
        ComponentKey(sub_component_type="decode"): 2,
    }


# ---------------------------------------------------------------------------
# Budget starvation (Q1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_starvation_emits_at_most_zero_no_at_least():
    # p_gpu=2, d_gpu=2, min_endpoint=1 → min_total=4; budget=3 (insufficient).
    config = _config(max_gpu_budget=3, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_disagg(p_gpu=2, d_gpu=2)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    # Expect AT_MOST=0 for both; no AT_LEAST anywhere.
    at_most = [
        (t.sub_component_type, t.replicas)
        for t in resp.override.targets
        if t.type == OverrideType.AT_MOST
    ]
    at_least = [t for t in resp.override.targets if t.type == OverrideType.AT_LEAST]
    assert ("prefill", 0) in at_most
    assert ("decode", 0) in at_most
    assert at_least == []


@pytest.mark.asyncio
async def test_budget_starvation_merge_yields_zero_replicas():
    config = _config(max_gpu_budget=3, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_disagg(p_gpu=2, d_gpu=2)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    plugin_pr = PluginResult(
        plugin_id="budget", priority=1, result=resp.override, final=False
    )
    upstream = ScalingProposal(
        targets=[
            ComponentTarget(sub_component_type="prefill", replicas=5, type=OverrideType.SET),
            ComponentTarget(sub_component_type="decode", replicas=5, type=OverrideType.SET),
        ]
    )
    outcome = _merge_with_upstream(plugin_pr, upstream)
    assert _replicas_by_key(outcome.proposal) == {
        ComponentKey(sub_component_type="prefill"): 0,
        ComponentKey(sub_component_type="decode"): 0,
    }


# ---------------------------------------------------------------------------
# scaling_in_progress: freeze to current
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scaling_in_progress_freezes_prefill_only():
    config = _config(max_gpu_budget=-1, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(_FakeOrchestrator(_caps_disagg()), config)  # type: ignore[arg-type]
    # Prefill is scaling (expected != ready); decode is stable.
    plugin.prime_tick(
        WorkerCounts(
            ready_num_prefill=2,
            expected_num_prefill=3,  # mismatch
            ready_num_decode=1,
            expected_num_decode=1,  # stable
        )
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    plugin_pr = PluginResult(
        plugin_id="budget", priority=1, result=resp.override, final=False
    )
    # Upstream wants prefill=5, decode=5. Should freeze prefill at 2, no
    # freeze on decode.
    upstream = ScalingProposal(
        targets=[
            ComponentTarget(sub_component_type="prefill", replicas=5, type=OverrideType.SET),
            ComponentTarget(sub_component_type="decode", replicas=5, type=OverrideType.SET),
        ]
    )
    outcome = _merge_with_upstream(plugin_pr, upstream)
    assert _replicas_by_key(outcome.proposal) == {
        ComponentKey(sub_component_type="prefill"): 2,  # frozen at current
        ComponentKey(sub_component_type="decode"): 5,  # not frozen, passthrough
    }


# ---------------------------------------------------------------------------
# Mode-specific: agg (only decode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agg_mode_only_decode_component():
    config = _config(mode="agg", max_gpu_budget=10, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_agg(d_gpu=2)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    sub_types = {t.sub_component_type for t in resp.override.targets}
    assert sub_types == {"decode"}
    # For agg with min_endpoint=1, d_gpu=2: ceiling = (10 - 0) // 2 = 5
    # (p_gpu is 0 since prefill caps absent).
    at_most = [t for t in resp.override.targets if t.type == OverrideType.AT_MOST]
    assert at_most[0].replicas == 5


# ---------------------------------------------------------------------------
# No capabilities → Accept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_capabilities_emits_at_least_only():
    """Without capabilities the plugin can't compute GPU-count ceilings
    but can still emit ``AT_LEAST(min_endpoint)`` per mode-indicated
    component. (Mode-based `_has_component` follows PSM; capabilities
    are a secondary source.)"""
    config = _config(max_gpu_budget=10, min_endpoint=1)  # default mode = disagg
    plugin = BuiltinBudgetConstrain(_FakeOrchestrator(None), config)  # type: ignore[arg-type]
    resp = await plugin.Constrain(ConstrainStageRequest())
    assert resp.result_kind == "override"
    at_least = [
        (t.sub_component_type, t.replicas)
        for t in resp.override.targets
        if t.type == OverrideType.AT_LEAST
    ]
    at_most = [t for t in resp.override.targets if t.type == OverrideType.AT_MOST]
    assert sorted(at_least) == [("decode", 1), ("prefill", 1)]
    # No AT_MOST — can't compute without p_gpu / d_gpu.
    assert at_most == []


# ---------------------------------------------------------------------------
# Upstream proposal untouched when within ceiling + above floor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_passthrough_when_upstream_within_bounds():
    config = _config(max_gpu_budget=20, min_endpoint=1)
    plugin = BuiltinBudgetConstrain(
        _FakeOrchestrator(_caps_disagg(p_gpu=2, d_gpu=1)), config  # type: ignore[arg-type]
    )
    resp = await plugin.Constrain(ConstrainStageRequest())
    plugin_pr = PluginResult(
        plugin_id="budget", priority=1, result=resp.override, final=False
    )
    # ceiling_p = (20 - 1) // 2 = 9; ceiling_d = (20 - 2) // 1 = 18
    # Upstream wants (3, 5) — within both floors and ceilings.
    upstream = ScalingProposal(
        targets=[
            ComponentTarget(sub_component_type="prefill", replicas=3, type=OverrideType.SET),
            ComponentTarget(sub_component_type="decode", replicas=5, type=OverrideType.SET),
        ]
    )
    outcome = _merge_with_upstream(plugin_pr, upstream)
    assert _replicas_by_key(outcome.proposal) == {
        ComponentKey(sub_component_type="prefill"): 3,
        ComponentKey(sub_component_type="decode"): 5,
    }
