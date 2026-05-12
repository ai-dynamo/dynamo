# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for type_aware_merge basic paths.

Covers the non-short-circuit, non-CONSTRAIN cases:
- baseline passthrough / AcceptResult passthrough
- SET recommendation (single + priority tiebreak)
- AT_LEAST floor (single + max of multi)
- AT_MOST ceiling (single + min of multi)
- clamp ordering when floor > ceiling
- SET clamped by floor / ceiling
- multi-component independent buckets
- multi-pool (component_name) independent buckets
- replicas=None ComponentTarget skipped
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

PREFILL = ComponentKey(sub_component_type="prefill")
DECODE = ComponentKey(sub_component_type="decode")
POOL_A = ComponentKey(sub_component_type="prefill", component_name="pool-A")
POOL_B = ComponentKey(sub_component_type="prefill", component_name="pool-B")


def _pr(plugin_id, priority, targets, final=False):
    return PluginResult(
        plugin_id=plugin_id,
        priority=priority,
        result=OverrideResult(targets=list(targets)),
        final=final,
    )


def _ct(sub_component_type, type_, replicas, component_name=None):
    return ComponentTarget(
        sub_component_type=sub_component_type,
        component_name=component_name,
        type=type_,
        replicas=replicas,
    )


def _replicas_by_key(outcome: MergeOutcome) -> dict[ComponentKey, int]:
    assert outcome.proposal is not None
    out: dict[ComponentKey, int] = {}
    for t in outcome.proposal.targets:
        key = ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        )
        assert t.replicas is not None
        out[key] = t.replicas
    return out


def test_empty_plugins_passes_through_baseline():
    out = type_aware_merge([], {PREFILL: 5})
    assert out.short_circuited is False
    assert _replicas_by_key(out) == {PREFILL: 5}
    assert out.used_final_from == ""
    assert out.set_dropped == []


def test_accept_only_passes_through_baseline():
    out = type_aware_merge(
        [PluginResult(plugin_id="p1", priority=100, result=AcceptResult())],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 5}


def test_single_set_wins_over_baseline():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)])],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 8}


def test_multi_set_priority_smallest_wins():
    # p2 has priority=50 (smaller number = higher priority) -> 10 wins
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.SET, 10)]),
        ],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 10}


def test_single_at_least_raises_floor_above_baseline():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.AT_LEAST, 6)])],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 6}


def test_multi_at_least_takes_max():
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.AT_LEAST, 4)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.AT_LEAST, 7)]),
        ],
        {PREFILL: 3},
    )
    assert _replicas_by_key(out) == {PREFILL: 7}


def test_single_at_most_lowers_ceiling_below_baseline():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.AT_MOST, 4)])],
        {PREFILL: 10},
    )
    assert _replicas_by_key(out) == {PREFILL: 4}


def test_multi_at_most_takes_min():
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.AT_MOST, 8)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.AT_MOST, 5)]),
        ],
        {PREFILL: 10},
    )
    assert _replicas_by_key(out) == {PREFILL: 5}


def test_floor_wins_when_floor_above_ceiling():
    # max(floor, min(ceiling, rec)) => floor wins because clamp is outer max.
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.AT_LEAST, 6)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.AT_MOST, 4)]),
        ],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 6}


def test_set_raised_by_at_least_floor():
    # SET=4 below AT_LEAST=6 -> floor pulls it up to 6.
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.SET, 4)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.AT_LEAST, 6)]),
        ],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 6}


def test_set_capped_by_at_most_ceiling():
    # SET=12 above AT_MOST=8 -> ceiling pulls it down to 8.
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.SET, 12)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.AT_MOST, 8)]),
        ],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 8}


def test_multi_component_independent_buckets():
    out = type_aware_merge(
        [
            _pr(
                "p1",
                100,
                [
                    _ct("prefill", OverrideType.SET, 8),
                    _ct("decode", OverrideType.SET, 4),
                ],
            )
        ],
        {PREFILL: 5, DECODE: 3},
    )
    assert _replicas_by_key(out) == {PREFILL: 8, DECODE: 4}


def test_component_name_creates_separate_buckets():
    out = type_aware_merge(
        [
            _pr(
                "p1",
                100,
                [
                    _ct("prefill", OverrideType.SET, 8, component_name="pool-A"),
                    _ct("prefill", OverrideType.SET, 4, component_name="pool-B"),
                ],
            )
        ],
        {POOL_A: 5, POOL_B: 3},
    )
    assert _replicas_by_key(out) == {POOL_A: 8, POOL_B: 4}


def test_unset_replicas_skipped_falls_back_to_baseline():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.SET, None)])],
        {PREFILL: 5},
    )
    assert _replicas_by_key(out) == {PREFILL: 5}


def test_proposal_source_is_merged_and_no_final_used():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)])],
        {PREFILL: 5},
    )
    assert out.proposal is not None
    assert out.proposal.source == "merged"
    assert out.used_final_from == ""
    assert out.set_dropped == []


def test_baseline_only_key_appears_in_output():
    # plugin touches prefill; decode only in baseline -> both present.
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)])],
        {PREFILL: 5, DECODE: 3},
    )
    assert _replicas_by_key(out) == {PREFILL: 8, DECODE: 3}


def test_empty_plugins_and_empty_baseline_emits_empty_proposal():
    out = type_aware_merge([], {})
    assert out.short_circuited is False
    assert out.proposal is not None
    assert out.proposal.targets == []
