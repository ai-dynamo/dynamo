# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``MergeOutcome.clamped`` population.

Extends existing ``type_aware_merge`` tests to verify the new
``clamped`` field accurately records which (key, direction, source)
events should drive RECONCILE/CONSTRAIN clamp counters.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.merge import (
    ComponentKey,
    MergeOutcome,
    PluginResult,
    type_aware_merge,
)
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    OverrideResult,
    OverrideType,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


PREFILL = ComponentKey(sub_component_type="prefill", component_name="worker_a")


def _override(plugin_id, priority, override_type, replicas):
    return PluginResult(
        plugin_id=plugin_id,
        priority=priority,
        result=OverrideResult(
            targets=[
                ComponentTarget(
                    sub_component_type="prefill",
                    component_name="worker_a",
                    replicas=replicas,
                    type=override_type,
                )
            ]
        ),
        final=False,
    )


# ---------------------------------------------------------------------------
# Empty clamp list when no clamping happens
# ---------------------------------------------------------------------------


def test_clamped_empty_when_no_override_present():
    outcome = type_aware_merge([], {PREFILL: 2}, set_allowed=True)
    assert outcome.clamped == []


def test_clamped_empty_when_set_matches_bounds():
    """SET=4 between AT_LEAST=2 and AT_MOST=6 — no clamp fires."""
    outcome = type_aware_merge(
        [
            _override("setter", 1, OverrideType.SET, 4),
            _override("floor", 2, OverrideType.AT_LEAST, 2),
            _override("ceiling", 3, OverrideType.AT_MOST, 6),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 4
    assert outcome.clamped == []


# ---------------------------------------------------------------------------
# Floor clamp (AT_LEAST raised recommendation)
# ---------------------------------------------------------------------------


def test_clamped_records_floor_when_at_least_raises_recommendation():
    """SET=1, AT_LEAST=5 → result=5, clamped=[(key, 'floor', 'floor_plugin')]."""
    outcome = type_aware_merge(
        [
            _override("set_plugin", 1, OverrideType.SET, 1),
            _override("floor_plugin", 2, OverrideType.AT_LEAST, 5),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 5
    assert len(outcome.clamped) == 1
    key, direction, source = outcome.clamped[0]
    assert key == PREFILL
    assert direction == "floor"
    assert source == "floor_plugin"


def test_clamped_floor_uses_winning_at_least_source():
    """Multiple AT_LEAST — the highest value wins; source labels that plugin."""
    outcome = type_aware_merge(
        [
            _override("set_plugin", 1, OverrideType.SET, 1),
            _override("weak_floor", 2, OverrideType.AT_LEAST, 2),
            _override("strong_floor", 3, OverrideType.AT_LEAST, 7),
            _override("mid_floor", 4, OverrideType.AT_LEAST, 4),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 7
    assert len(outcome.clamped) == 1
    assert outcome.clamped[0][2] == "strong_floor"


def test_clamped_floor_fires_even_when_baseline_recommendation():
    """No SET; baseline=1; AT_LEAST=4 → floor clamp still recorded."""
    outcome = type_aware_merge(
        [_override("floor", 1, OverrideType.AT_LEAST, 4)],
        {PREFILL: 1},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 4
    assert outcome.clamped == [(PREFILL, "floor", "floor")]


# ---------------------------------------------------------------------------
# Ceiling clamp (AT_MOST lowered recommendation)
# ---------------------------------------------------------------------------


def test_clamped_records_ceiling_when_at_most_lowers_recommendation():
    """SET=10, AT_MOST=3 → result=3, clamped=[(key, 'ceiling', ...)]."""
    outcome = type_aware_merge(
        [
            _override("set_plugin", 1, OverrideType.SET, 10),
            _override("budget", 2, OverrideType.AT_MOST, 3),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 3
    assert len(outcome.clamped) == 1
    key, direction, source = outcome.clamped[0]
    assert direction == "ceiling"
    assert source == "budget"


def test_clamped_ceiling_uses_tightest_at_most_source():
    """Multiple AT_MOST — the lowest value wins; source labels that plugin."""
    outcome = type_aware_merge(
        [
            _override("set_plugin", 1, OverrideType.SET, 10),
            _override("loose_budget", 2, OverrideType.AT_MOST, 8),
            _override("tight_budget", 3, OverrideType.AT_MOST, 4),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 4
    assert len(outcome.clamped) == 1
    assert outcome.clamped[0][2] == "tight_budget"


# ---------------------------------------------------------------------------
# Both floor + ceiling fire when they clamp simultaneously
# ---------------------------------------------------------------------------


def test_clamped_records_only_winning_direction_when_floor_exceeds_ceiling():
    """Degenerate case: AT_LEAST=7 > AT_MOST=3. Spec: floor wins
    (result=7). The ceiling ``tried`` to lower the recommendation but
    the floor overrode it, so the net effect is only a floor clamp.
    Only the direction that actually changed the output vs
    recommendation is recorded — dashboards should show "floor clamped
    this component" but not a ceiling that was itself overridden.
    """
    outcome = type_aware_merge(
        [
            _override("set_plugin", 1, OverrideType.SET, 1),
            _override("high_floor", 2, OverrideType.AT_LEAST, 7),
            _override("tight_budget", 3, OverrideType.AT_MOST, 3),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.proposal.targets[0].replicas == 7
    assert len(outcome.clamped) == 1
    assert outcome.clamped[0][1] == "floor"
    assert outcome.clamped[0][2] == "high_floor"


# ---------------------------------------------------------------------------
# Multi-key clamps are reported per-key
# ---------------------------------------------------------------------------


DECODE = ComponentKey(sub_component_type="decode", component_name="worker_b")


def test_clamped_reports_per_component_independently():
    def override_for(key, plugin_id, priority, ot, replicas):
        return PluginResult(
            plugin_id=plugin_id,
            priority=priority,
            result=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type=key.sub_component_type,
                        component_name=key.component_name,
                        replicas=replicas,
                        type=ot,
                    )
                ]
            ),
            final=False,
        )

    outcome = type_aware_merge(
        [
            override_for(PREFILL, "set_p", 1, OverrideType.SET, 1),
            override_for(PREFILL, "floor_p", 2, OverrideType.AT_LEAST, 5),
            override_for(DECODE, "set_d", 1, OverrideType.SET, 10),
            override_for(DECODE, "cap_d", 2, OverrideType.AT_MOST, 4),
        ],
        {PREFILL: 0, DECODE: 0},
        set_allowed=True,
    )
    keys_clamped = {(k.sub_component_type, d) for k, d, _ in outcome.clamped}
    assert keys_clamped == {("prefill", "floor"), ("decode", "ceiling")}


# ---------------------------------------------------------------------------
# Short-circuit (REJECT) keeps clamped empty (no merge happened)
# ---------------------------------------------------------------------------


def test_short_circuit_leaves_clamped_empty():
    from dynamo.planner.plugins.types import RejectResult

    outcome = type_aware_merge(
        [
            PluginResult(
                plugin_id="nope",
                priority=1,
                result=RejectResult(reason="no"),
                final=False,
            ),
        ],
        {PREFILL: 0},
        set_allowed=True,
    )
    assert outcome.short_circuited is True
    assert outcome.clamped == []
