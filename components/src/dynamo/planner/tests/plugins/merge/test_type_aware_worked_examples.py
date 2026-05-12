# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worked example verbatim assertions.

This file is the **source of truth** for type_aware_merge behaviour. The 9
cases below mirror the worked example table in the design doc PROPOSE
section verbatim — any edit to that table MUST come with a matching
edit here (and vice versa). Test IDs match the ``case_name`` so CI output
points directly at the offending case.

Helpers ``PR`` / ``OR`` / ``CT`` / ``key`` are named to make each row read
like the source table:

    PR("p1", 100, OR([CT("prefill", SET, 8)]))
    key("prefill", "pool-A")
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
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

SET = OverrideType.SET
AT_LEAST = OverrideType.AT_LEAST
AT_MOST = OverrideType.AT_MOST


def PR(plugin_id, priority, result, final=False):
    return PluginResult(
        plugin_id=plugin_id, priority=priority, result=result, final=final
    )


def OR(targets):
    return OverrideResult(targets=list(targets))


def CT(sub_component_type, *args):
    """``CT("prefill", SET, 5)`` or ``CT("prefill", "pool-A", SET, 5)``."""
    if len(args) == 2:
        type_, replicas = args
        return ComponentTarget(
            sub_component_type=sub_component_type,
            type=type_,
            replicas=replicas,
        )
    if len(args) == 3:
        component_name, type_, replicas = args
        return ComponentTarget(
            sub_component_type=sub_component_type,
            component_name=component_name,
            type=type_,
            replicas=replicas,
        )
    raise TypeError(
        f"CT expected 2 or 3 positional args after sub_component_type, got {len(args)}"
    )


def key(sub_component_type, component_name=None):
    return ComponentKey(
        sub_component_type=sub_component_type, component_name=component_name
    )


WORKED_EXAMPLES = [
    # (case_name, plugin_results, baseline, expected_replicas_by_key)
    # -- Single-component rows --
    (
        "only_baseline",
        [],
        {key("prefill"): 5},
        {key("prefill"): 5},
    ),
    (
        "only_set",
        [PR("p1", 100, OR([CT("prefill", SET, 8)]))],
        {key("prefill"): 5},
        {key("prefill"): 8},
    ),
    (
        "set_priority_wins",
        [
            PR("p1", 100, OR([CT("prefill", SET, 8)])),
            PR("p2", 50, OR([CT("prefill", SET, 10)])),
        ],
        {key("prefill"): 5},
        # p2 priority=50 (smaller number = higher precedence) -> 10 wins.
        {key("prefill"): 10},
    ),
    (
        "set_with_at_least_floor",
        [
            PR("p1", 100, OR([CT("prefill", SET, 4)])),
            PR("p2", 50, OR([CT("prefill", AT_LEAST, 6)])),
        ],
        {key("prefill"): 5},
        # SET=4 pulled up to floor=6.
        {key("prefill"): 6},
    ),
    (
        "set_with_at_most_ceiling",
        [
            PR("p1", 100, OR([CT("prefill", SET, 12)])),
            PR("p2", 50, OR([CT("prefill", AT_MOST, 8)])),
        ],
        {key("prefill"): 5},
        # SET=12 clamped down to ceiling=8.
        {key("prefill"): 8},
    ),
    # -- Multi-component rows --
    (
        "multi_component_independent",
        [
            PR(
                "p1",
                100,
                OR([CT("prefill", SET, 8), CT("decode", SET, 4)]),
            )
        ],
        {key("prefill"): 5, key("decode"): 3},
        {key("prefill"): 8, key("decode"): 4},
    ),
    (
        "multi_component_mixed_types",
        [
            PR(
                "p1",
                100,
                OR([CT("prefill", SET, 8), CT("decode", AT_MOST, 6)]),
            ),
            PR("p2", 50, OR([CT("decode", SET, 10)])),
        ],
        {key("prefill"): 5, key("decode"): 3},
        # decode SET=10 clamped down to AT_MOST=6.
        {key("prefill"): 8, key("decode"): 6},
    ),
    # -- Hierarchical pools (component_name disambiguates buckets) --
    (
        "hierarchical_pools",
        [
            PR(
                "p1",
                100,
                OR(
                    [
                        CT("prefill", "pool-A", SET, 8),
                        CT("prefill", "pool-B", SET, 4),
                    ]
                ),
            )
        ],
        {key("prefill", "pool-A"): 5, key("prefill", "pool-B"): 3},
        {key("prefill", "pool-A"): 8, key("prefill", "pool-B"): 4},
    ),
    # -- final verbatim override --
    (
        "final_override_completely",
        [
            PR("p1", 100, OR([CT("prefill", SET, 8)]), final=True),
            PR("p2", 50, OR([CT("prefill", SET, 10)])),
            PR("p3", 30, OR([CT("prefill", AT_MOST, 4)])),
        ],
        {key("prefill"): 5},
        # p1 final=True: its target wins verbatim; p2/p3 fully discarded.
        {key("prefill"): 8},
    ),
]


@pytest.mark.parametrize(
    "case_name,plugin_results,baseline,expected",
    WORKED_EXAMPLES,
    ids=[c[0] for c in WORKED_EXAMPLES],
)
def test_worked_example(case_name, plugin_results, baseline, expected):
    out = type_aware_merge(plugin_results, baseline, set_allowed=True)
    assert (
        out.proposal is not None
    ), f"case={case_name}: proposal unexpectedly None (short_circuited={out.short_circuited})"
    actual = {
        ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        ): t.replicas
        for t in out.proposal.targets
    }
    assert actual == expected, (
        f"case={case_name}: expected={expected}, got={actual}"
    )


def test_worked_examples_count_matches_main_doc():
    # Tripwire: main doc v11 PROPOSE table has exactly 9 worked examples.
    # If this count drifts, the doc and test are no longer in lock-step.
    assert len(WORKED_EXAMPLES) == 9
