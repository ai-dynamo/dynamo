# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for type_aware_merge REJECT short-circuit + final paths.

REJECT matrix (REJECT > final priority):
- single REJECT → short_circuited, proposal=None, reason includes plugin_id
- REJECT + other SET → still short-circuits
- REJECT + final OverrideResult → still short-circuits
- multiple REJECTs → short_circuit_reason reflects the first one

final matrix (priority rule for PROPOSE/RECONCILE):
- single final SET → that plugin's targets become the proposal verbatim
- multiple finals → priority-smallest wins
- final + non-final bounds → non-final entries discarded
- final with AT_LEAST only → proposal carries that AT_LEAST target verbatim
- final in CONSTRAIN (set_allowed=False) → SET dropped, set_dropped recorded,
  used_final_from still set (final remains authoritative for non-SET targets)
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.merge import (
    ComponentKey,
    PluginResult,
    type_aware_merge,
)
from dynamo.planner.plugins.types import (
    ComponentTarget,
    OverrideResult,
    OverrideType,
    RejectResult,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PREFILL = ComponentKey(sub_component_type="prefill")
DECODE = ComponentKey(sub_component_type="decode")


def _reject(plugin_id, priority, reason):
    return PluginResult(
        plugin_id=plugin_id,
        priority=priority,
        result=RejectResult(reason=reason),
    )


def _override(plugin_id, priority, targets, final=False):
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


# ---------------------------------------------------------------------------
# REJECT short-circuit matrix
# ---------------------------------------------------------------------------


def test_single_reject_short_circuits():
    out = type_aware_merge(
        [_reject("p1", 100, "over-capacity")],
        {PREFILL: 5},
    )
    assert out.short_circuited is True
    assert out.proposal is None
    assert "p1" in out.short_circuit_reason
    assert "over-capacity" in out.short_circuit_reason


def test_reject_with_other_set_still_short_circuits():
    out = type_aware_merge(
        [
            _override("p1", 100, [_ct("prefill", OverrideType.SET, 8)]),
            _reject("p2", 50, "nope"),
        ],
        {PREFILL: 5},
    )
    assert out.short_circuited is True
    assert out.proposal is None


def test_reject_outranks_final():
    # REJECT > final priority even when final is priority-small.
    out = type_aware_merge(
        [
            _override("p1", 10, [_ct("prefill", OverrideType.SET, 8)], final=True),
            _reject("p2", 100, "safety veto"),
        ],
        {PREFILL: 5},
    )
    assert out.short_circuited is True
    assert out.proposal is None
    assert out.used_final_from == ""


def test_multiple_rejects_reports_first_encountered():
    out = type_aware_merge(
        [
            _reject("p1", 100, "first"),
            _reject("p2", 50, "second"),
        ],
        {PREFILL: 5},
    )
    assert out.short_circuited is True
    assert "p1" in out.short_circuit_reason
    assert "first" in out.short_circuit_reason
    assert "p2" not in out.short_circuit_reason


# ---------------------------------------------------------------------------
# final priority matrix
# ---------------------------------------------------------------------------


def test_single_final_overrides_all():
    # p1 final SET=8; p2 non-final SET=10 (priority-smaller) discarded.
    out = type_aware_merge(
        [
            _override("p1", 100, [_ct("prefill", OverrideType.SET, 8)], final=True),
            _override("p2", 50, [_ct("prefill", OverrideType.SET, 10)]),
        ],
        {PREFILL: 5},
    )
    assert out.proposal is not None
    assert out.used_final_from == "p1"
    assert out.proposal.source == "p1"
    assert len(out.proposal.targets) == 1
    assert out.proposal.targets[0].replicas == 8


def test_multiple_finals_priority_smallest_wins():
    # Both final; p2 has smaller priority (higher precedence) -> its SET wins.
    out = type_aware_merge(
        [
            _override("p1", 100, [_ct("prefill", OverrideType.SET, 8)], final=True),
            _override("p2", 50, [_ct("prefill", OverrideType.SET, 10)], final=True),
        ],
        {PREFILL: 5},
    )
    assert out.proposal is not None
    assert out.used_final_from == "p2"
    assert out.proposal.targets[0].replicas == 10


def test_final_discards_non_final_bounds():
    # final's OverrideResult is taken verbatim; non-final AT_LEAST/AT_MOST
    # from other plugins are fully ignored (no clamp).
    out = type_aware_merge(
        [
            _override("p1", 100, [_ct("prefill", OverrideType.SET, 8)], final=True),
            _override("p2", 50, [_ct("prefill", OverrideType.AT_LEAST, 20)]),
            _override("p3", 30, [_ct("prefill", OverrideType.AT_MOST, 4)]),
        ],
        {PREFILL: 5},
    )
    assert out.proposal is not None
    assert out.used_final_from == "p1"
    assert len(out.proposal.targets) == 1
    assert out.proposal.targets[0].replicas == 8
    assert out.proposal.targets[0].type == OverrideType.SET


def test_final_with_at_least_only_preserves_type():
    # ScalingProposal.ComponentTarget.type is unused downstream but the
    # verbatim-passthrough contract should preserve whatever the final
    # plugin emitted.
    out = type_aware_merge(
        [
            _override(
                "p1",
                100,
                [_ct("prefill", OverrideType.AT_LEAST, 7)],
                final=True,
            ),
        ],
        {PREFILL: 5},
    )
    assert out.proposal is not None
    assert out.used_final_from == "p1"
    assert len(out.proposal.targets) == 1
    assert out.proposal.targets[0].type == OverrideType.AT_LEAST
    assert out.proposal.targets[0].replicas == 7


def test_final_in_constrain_drops_set_but_final_still_applied():
    # CONSTRAIN + final containing SET + AT_MOST:
    # - SET prefill dropped and recorded
    # - AT_MOST decode preserved
    # - used_final_from set (final still authoritative for non-SET entries)
    out = type_aware_merge(
        [
            _override(
                "p1",
                100,
                [
                    _ct("prefill", OverrideType.SET, 8),
                    _ct("decode", OverrideType.AT_MOST, 4),
                ],
                final=True,
            ),
        ],
        {PREFILL: 3, DECODE: 2},
        set_allowed=False,
    )
    assert out.proposal is not None
    assert out.used_final_from == "p1"
    assert out.set_dropped == [PREFILL]

    remaining = [
        (t.sub_component_type, t.type, t.replicas) for t in out.proposal.targets
    ]
    assert ("decode", OverrideType.AT_MOST, 4) in remaining
    # prefill SET was the dropped one; nothing else for prefill in final path
    assert all(t.sub_component_type != "prefill" for t in out.proposal.targets)
