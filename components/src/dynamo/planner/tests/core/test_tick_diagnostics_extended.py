# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plugin-era ``TickDiagnostics`` fields.

The extension adds three plugin-aware fields:
- plugin_overrides: list[tuple[str, str, str, str, int]]
- reconcile_reasons: dict[str, str]
- held_over_plugins: list[str]

All three default to empty collections so PSM-path callers that never
touch them still produce a byte-identical ``TickDiagnostics()`` value.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from dynamo.planner.core.types import TickDiagnostics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_are_empty_collections():
    d = TickDiagnostics()
    assert d.plugin_overrides == []
    assert d.reconcile_reasons == {}
    assert d.held_over_plugins == []


def test_default_collections_are_not_shared_across_instances():
    """Regression guard: ``field(default_factory=list)`` prevents the
    classic mutable-default aliasing bug. This test makes the invariant
    explicit so a future refactor that swaps to a bare default doesn't
    slip through."""
    a = TickDiagnostics()
    b = TickDiagnostics()
    a.plugin_overrides.append(("p1", "propose", "SET", "prefill/w1", 3))
    a.reconcile_reasons["prefill/w1"] = "set_by_p1"
    a.held_over_plugins.append("p2")
    assert b.plugin_overrides == []
    assert b.reconcile_reasons == {}
    assert b.held_over_plugins == []


# ---------------------------------------------------------------------------
# Field population + shape checks
# ---------------------------------------------------------------------------


def test_plugin_overrides_accepts_tuple_shape():
    d = TickDiagnostics()
    d.plugin_overrides.append(("my_plugin", "propose", "SET", "decode/w1", 5))
    d.plugin_overrides.append(("other_plugin", "constrain", "AT_MOST", "", 8))
    # REJECT uses -1 placeholder per the contract documented on the field.
    d.plugin_overrides.append(("safe", "reconcile", "REJECT", "", -1))

    assert len(d.plugin_overrides) == 3
    plugin_ids = [o[0] for o in d.plugin_overrides]
    stages = [o[1] for o in d.plugin_overrides]
    types_ = [o[2] for o in d.plugin_overrides]
    assert plugin_ids == ["my_plugin", "other_plugin", "safe"]
    assert stages == ["propose", "constrain", "reconcile"]
    assert types_ == ["SET", "AT_MOST", "REJECT"]


def test_reconcile_reasons_accepts_component_key_mapping():
    d = TickDiagnostics()
    d.reconcile_reasons["prefill/worker_a"] = "set_by_budget_constrain"
    d.reconcile_reasons["decode/worker_a"] = "clamped_to_floor"
    d.reconcile_reasons["decode/worker_b"] = "passthrough"

    assert d.reconcile_reasons["prefill/worker_a"] == "set_by_budget_constrain"
    assert d.reconcile_reasons["decode/worker_a"] == "clamped_to_floor"
    assert d.reconcile_reasons["decode/worker_b"] == "passthrough"


def test_held_over_plugins_accepts_plugin_id_list():
    d = TickDiagnostics()
    d.held_over_plugins.extend(["slow_predictor", "bursty_propose"])
    assert d.held_over_plugins == ["slow_predictor", "bursty_propose"]


# ---------------------------------------------------------------------------
# asdict round-trip (for replay + dashboard serialisation)
# ---------------------------------------------------------------------------


def test_asdict_round_trip_preserves_new_fields():
    d = TickDiagnostics(
        load_decision_reason="no_change",
        plugin_overrides=[("p1", "propose", "SET", "decode/w", 3)],
        reconcile_reasons={"decode/w": "set_by_p1"},
        held_over_plugins=["p2"],
    )
    encoded = dataclasses.asdict(d)
    decoded = TickDiagnostics(**encoded)
    assert decoded == d


def test_asdict_empty_new_fields_round_trip():
    """PSM path never writes the new fields; round-trip must preserve
    the empty defaults without drifting to None or leaking."""
    d = TickDiagnostics(load_decision_reason="scale_up", estimated_itl_ms=12.3)
    encoded = dataclasses.asdict(d)
    decoded = TickDiagnostics(**encoded)
    assert decoded == d
    assert decoded.plugin_overrides == []
    assert decoded.reconcile_reasons == {}
    assert decoded.held_over_plugins == []


# ---------------------------------------------------------------------------
# Backward compatibility with PSM path
# ---------------------------------------------------------------------------


def test_psm_style_construction_still_works():
    """PSM call sites only pass existing numeric + reason fields; adding
    new fields with default_factory MUST NOT break those call sites."""
    d = TickDiagnostics(
        estimated_ttft_ms=12.3,
        estimated_itl_ms=4.5,
        predicted_num_req=100.0,
        predicted_isl=1500.0,
        predicted_osl=200.0,
        engine_rps_prefill=2.0,
        engine_rps_decode=0.4,
        throughput_lower_bound_prefill=2,
        throughput_lower_bound_decode=3,
        load_decision_reason="no_change",
        throughput_decision_reason="scale",
    )
    # Original fields preserved.
    assert d.estimated_ttft_ms == 12.3
    assert d.throughput_lower_bound_prefill == 2
    # New fields defaulted.
    assert d.plugin_overrides == []
    assert d.reconcile_reasons == {}
    assert d.held_over_plugins == []


def test_deepcopy_of_populated_diagnostics():
    """Some consumers (diagnostics_recorder, replay) deepcopy the
    TickDiagnostics to snapshot state. Verify the new fields survive."""
    d = TickDiagnostics(
        plugin_overrides=[("p", "propose", "SET", "k", 1)],
        reconcile_reasons={"k": "r"},
        held_over_plugins=["q"],
    )
    d2 = copy.deepcopy(d)
    assert d2 == d
    # Deep copy: mutating one must not affect the other.
    d2.plugin_overrides.clear()
    d2.reconcile_reasons.clear()
    d2.held_over_plugins.clear()
    assert d.plugin_overrides == [("p", "propose", "SET", "k", 1)]
    assert d.reconcile_reasons == {"k": "r"}
    assert d.held_over_plugins == ["q"]
