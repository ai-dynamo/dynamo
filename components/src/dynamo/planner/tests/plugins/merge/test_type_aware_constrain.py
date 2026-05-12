# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for type_aware_merge CONSTRAIN mode.

With ``set_allowed=False`` (CONSTRAIN stage):
- SET targets are silently dropped from the merge
- dropped keys are recorded in ``MergeOutcome.set_dropped`` for audit
- AT_LEAST / AT_MOST bounds merge normally

Register-time static rejection of CONSTRAIN-SET plugins is infeasible
(proto3 has no plugin-declared output-type metadata); drop + audit is
the only workable approach.
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


def test_single_set_dropped_and_baseline_passthrough():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)])],
        {PREFILL: 5},
        set_allowed=False,
    )
    assert _replicas_by_key(out) == {PREFILL: 5}
    assert out.set_dropped == [PREFILL]


def test_at_least_merges_normally_when_set_disallowed():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.AT_LEAST, 6)])],
        {PREFILL: 5},
        set_allowed=False,
    )
    # baseline=5 raised by floor=6
    assert _replicas_by_key(out) == {PREFILL: 6}
    assert out.set_dropped == []


def test_at_most_merges_normally_when_set_disallowed():
    out = type_aware_merge(
        [_pr("p1", 100, [_ct("prefill", OverrideType.AT_MOST, 4)])],
        {PREFILL: 10},
        set_allowed=False,
    )
    assert _replicas_by_key(out) == {PREFILL: 4}
    assert out.set_dropped == []


def test_mixed_set_and_bounds_drops_only_sets():
    # p1: prefill SET (dropped) + decode AT_MOST (kept)
    # p2: decode SET (dropped)
    # Expected: prefill = baseline 3; decode = baseline 2 clamped by AT_MOST=6 => 2
    out = type_aware_merge(
        [
            _pr(
                "p1",
                100,
                [
                    _ct("prefill", OverrideType.SET, 12),
                    _ct("decode", OverrideType.AT_MOST, 6),
                ],
            ),
            _pr("p2", 50, [_ct("decode", OverrideType.SET, 10)]),
        ],
        {PREFILL: 3, DECODE: 2},
        set_allowed=False,
    )
    assert _replicas_by_key(out) == {PREFILL: 3, DECODE: 2}
    # Drop order follows iteration: p1.prefill first, then p2.decode.
    assert out.set_dropped == [PREFILL, DECODE]


def test_duplicate_set_keys_recorded_per_plugin():
    # Same key SET from two plugins: both recorded in set_dropped so
    # orchestrator can bump the per-plugin Prometheus counter correctly.
    out = type_aware_merge(
        [
            _pr("p1", 100, [_ct("prefill", OverrideType.SET, 8)]),
            _pr("p2", 50, [_ct("prefill", OverrideType.SET, 10)]),
        ],
        {PREFILL: 5},
        set_allowed=False,
    )
    assert _replicas_by_key(out) == {PREFILL: 5}
    assert out.set_dropped == [PREFILL, PREFILL]
