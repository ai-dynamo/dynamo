# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis property-based tests for type_aware_merge (PR 4 sub-task 4-6).

Invariants (v11 main doc § 4.3):

1. **clamp correctness** — every output replicas value satisfies
   ``floor <= replicas <= ceiling``, where ``floor = max(AT_LEAST)`` and
   ``ceiling = min(AT_MOST)`` for that ComponentKey (default 0 / +inf).
   Exception: when ``floor > ceiling``, floor wins (clamp is
   ``max(floor, min(ceiling, rec))``), and replicas == floor.
2. **baseline preservation** — for any key that no plugin touches but is
   present in ``baseline``, the output replicas equal ``baseline[key]``.
3. **REJECT dominance** — adding any RejectResult to ``plugin_results``
   always produces ``short_circuited=True``, regardless of what other
   plugins say.
4. **final dominance** — adding a ``final=True`` OverrideResult with
   priority = 0 (lowest number, highest precedence) produces a proposal
   whose targets are that plugin's targets verbatim (no REJECT present).
5. **monotone AT_LEAST** — adding an extra AT_LEAST=k plugin cannot
   decrease any ComponentKey's result; per-key result is
   ``>= max(floor_before, k)``.
6. **idempotency** — duplicating a plugin's output (same targets, same
   priority) does not change the merge result.

Requires ``hypothesis`` (dev extra; see ``DEP-XXXX_PR4_Detailed_zh.md``
Q4 decision). The file is auto-skipped when hypothesis is unavailable.
"""

from __future__ import annotations

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------


component_types = st.sampled_from(["prefill", "decode"])
component_names = st.one_of(st.none(), st.sampled_from(["pool-A", "pool-B"]))
replicas_st = st.integers(min_value=0, max_value=50)
priority_st = st.integers(min_value=1, max_value=200)
plugin_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=6,
).map(lambda s: "p_" + s)


@st.composite
def component_targets(draw, allow_set=True):
    types = [OverrideType.AT_LEAST, OverrideType.AT_MOST]
    if allow_set:
        types.append(OverrideType.SET)
    return ComponentTarget(
        sub_component_type=draw(component_types),
        component_name=draw(component_names),
        replicas=draw(replicas_st),
        type=draw(st.sampled_from(types)),
    )


@st.composite
def override_plugin_results(draw, allow_final=True):
    targets = draw(st.lists(component_targets(), min_size=1, max_size=3))
    return PluginResult(
        plugin_id=draw(plugin_id_st),
        priority=draw(priority_st),
        result=OverrideResult(targets=targets),
        final=draw(st.booleans()) if allow_final else False,
    )


baseline_st = st.dictionaries(
    keys=st.builds(
        ComponentKey,
        sub_component_type=component_types,
        component_name=component_names,
    ),
    values=replicas_st,
    max_size=4,
)


def _floor_ceiling(plugin_results, key):
    at_least = []
    at_most = []
    for r in plugin_results:
        if not isinstance(r.result, OverrideResult):
            continue
        for t in r.result.targets:
            if t.replicas is None:
                continue
            tkey = ComponentKey(
                sub_component_type=t.sub_component_type,
                component_name=t.component_name,
            )
            if tkey != key:
                continue
            if t.type == OverrideType.AT_LEAST:
                at_least.append(t.replicas)
            elif t.type == OverrideType.AT_MOST:
                at_most.append(t.replicas)
    floor = max(at_least) if at_least else 0
    ceiling = min(at_most) if at_most else math.inf
    return floor, ceiling


# ----------------------------------------------------------------------------
# Invariant 1 — clamp correctness
# ----------------------------------------------------------------------------


@given(
    plugin_results=st.lists(
        override_plugin_results(allow_final=False), max_size=4
    ),
    baseline=baseline_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_clamp_correctness(plugin_results, baseline):
    out = type_aware_merge(plugin_results, baseline, set_allowed=True)
    assert out.proposal is not None
    for t in out.proposal.targets:
        key = ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        )
        floor, ceiling = _floor_ceiling(plugin_results, key)
        if floor <= ceiling:
            assert floor <= t.replicas <= ceiling, (
                f"key={key} replicas={t.replicas} not in [{floor}, {ceiling}]"
            )
        else:
            # floor > ceiling: outer max wins, replicas = floor
            assert t.replicas == floor


# ----------------------------------------------------------------------------
# Invariant 2 — baseline preservation when no plugin touches a key
# ----------------------------------------------------------------------------


@given(baseline=baseline_st)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_baseline_preservation_when_no_plugin(baseline):
    out = type_aware_merge([], baseline, set_allowed=True)
    assert out.proposal is not None
    by_key = {
        ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        ): t.replicas
        for t in out.proposal.targets
    }
    assert by_key == dict(baseline)


# ----------------------------------------------------------------------------
# Invariant 3 — REJECT dominance
# ----------------------------------------------------------------------------


@given(
    plugin_results=st.lists(
        override_plugin_results(allow_final=True), max_size=4
    ),
    baseline=baseline_st,
    reject_priority=priority_st,
    reject_reason=st.text(max_size=10),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_reject_always_short_circuits(
    plugin_results, baseline, reject_priority, reject_reason
):
    reject = PluginResult(
        plugin_id="rej",
        priority=reject_priority,
        result=RejectResult(reason=reject_reason),
    )
    out = type_aware_merge(
        list(plugin_results) + [reject], baseline, set_allowed=True
    )
    assert out.short_circuited is True
    assert out.proposal is None


# ----------------------------------------------------------------------------
# Invariant 4 — final dominance (priority=0, no REJECT present)
# ----------------------------------------------------------------------------


@given(
    other_results=st.lists(
        override_plugin_results(allow_final=False).filter(
            lambda pr: pr.priority > 0
        ),
        max_size=3,
    ),
    final_targets=st.lists(component_targets(allow_set=True), min_size=1, max_size=2),
    baseline=baseline_st,
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_final_at_priority_zero_dominates(other_results, final_targets, baseline):
    final_pr = PluginResult(
        plugin_id="final",
        priority=0,  # smallest number wins among finals
        result=OverrideResult(targets=final_targets),
        final=True,
    )
    out = type_aware_merge(
        list(other_results) + [final_pr], baseline, set_allowed=True
    )
    assert out.short_circuited is False
    assert out.used_final_from == "final"
    assert out.proposal is not None
    # Verbatim passthrough of final's targets.
    assert list(out.proposal.targets) == final_targets


# ----------------------------------------------------------------------------
# Invariant 5 — monotone AT_LEAST
# ----------------------------------------------------------------------------


@given(
    plugin_results=st.lists(
        override_plugin_results(allow_final=False), max_size=3
    ),
    extra_key=st.builds(
        ComponentKey,
        sub_component_type=component_types,
        component_name=component_names,
    ),
    extra_replicas=replicas_st,
    extra_priority=priority_st,
    baseline=baseline_st,
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_at_least_monotone(
    plugin_results, extra_key, extra_replicas, extra_priority, baseline
):
    before = type_aware_merge(plugin_results, baseline, set_allowed=True)
    extra = PluginResult(
        plugin_id="extra",
        priority=extra_priority,
        result=OverrideResult(
            targets=[
                ComponentTarget(
                    sub_component_type=extra_key.sub_component_type,
                    component_name=extra_key.component_name,
                    replicas=extra_replicas,
                    type=OverrideType.AT_LEAST,
                )
            ]
        ),
    )
    after = type_aware_merge(
        list(plugin_results) + [extra], baseline, set_allowed=True
    )
    assert before.proposal is not None
    assert after.proposal is not None
    before_map = {
        ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        ): t.replicas
        for t in before.proposal.targets
    }
    after_map = {
        ComponentKey(
            sub_component_type=t.sub_component_type,
            component_name=t.component_name,
        ): t.replicas
        for t in after.proposal.targets
    }
    # Every key shared by both proposals: after >= before.
    for key, replicas_before in before_map.items():
        if key in after_map:
            assert after_map[key] >= replicas_before, (
                f"AT_LEAST monotonicity violated at {key}: "
                f"before={replicas_before}, after={after_map[key]}"
            )
    # The extra_key must satisfy after_replicas >= extra_replicas (floor pull-up).
    assert after_map[extra_key] >= extra_replicas


# ----------------------------------------------------------------------------
# Invariant 6 — idempotency of duplicated plugin
# ----------------------------------------------------------------------------


@given(
    plugin_results=st.lists(
        override_plugin_results(allow_final=False), min_size=1, max_size=3
    ),
    baseline=baseline_st,
    dup_index=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_duplicate_plugin_result_is_idempotent(
    plugin_results, baseline, dup_index
):
    idx = dup_index % len(plugin_results)
    duplicated = list(plugin_results) + [plugin_results[idx]]
    base_out = type_aware_merge(plugin_results, baseline, set_allowed=True)
    dup_out = type_aware_merge(duplicated, baseline, set_allowed=True)
    assert base_out.proposal is not None
    assert dup_out.proposal is not None
    base_map = {
        (t.sub_component_type, t.component_name): t.replicas
        for t in base_out.proposal.targets
    }
    dup_map = {
        (t.sub_component_type, t.component_name): t.replicas
        for t in dup_out.proposal.targets
    }
    assert base_map == dup_map
