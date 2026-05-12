# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LOAD/THROUGHPUT decision state enum extensions.

Enum.states is fixed at construction time; these tests guard against
accidental removal or reordering that would break scrapers.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry, Enum

from dynamo.planner.monitoring.planner_metrics import (
    LOAD_DECISION_STATES,
    THROUGHPUT_DECISION_STATES,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# The original v1 states MUST remain in the list and MUST remain at the
# same positions so scrapers reading older label sets keep working.
_V1_LOAD_STATES = [
    "unset",
    "disabled",
    "no_fpm_data",
    "scaling_in_progress",
    "worker_count_mismatch",
    "insufficient_data",
    "no_change",
    "scale_up",
    "scale_down",
    "scale_down_capped_by_throughput",
    # Upstream main added this after the original v1 list but before the
    # plugin-era additions. Append-only contract still honoured.
    "scale_down_refused_consolidation",
]

_V1_THROUGHPUT_STATES = [
    "unset",
    "disabled",
    "no_traffic_data",
    "predict_failed",
    "model_not_ready",
    "set_lower_bound",
    "scale",
]

# Plugin-era additions — appended in this order.
_PLUGIN_LOAD_ADDITIONS = [
    "override_by_user_plugin",
    "reconcile_clamped_to_floor",
    "reconcile_clamped_to_ceiling",
    "held_over",
    "rejected_by_plugin",
]

_PLUGIN_THROUGHPUT_ADDITIONS = [
    "override_by_user_plugin",
    "held_over",
    "circuit_open",
    "rejected_by_plugin",
]


def test_v1_load_states_preserved_in_original_order():
    assert LOAD_DECISION_STATES[: len(_V1_LOAD_STATES)] == _V1_LOAD_STATES


def test_v1_throughput_states_preserved_in_original_order():
    assert (
        THROUGHPUT_DECISION_STATES[: len(_V1_THROUGHPUT_STATES)]
        == _V1_THROUGHPUT_STATES
    )


def test_v11_load_additions_appended_in_order():
    assert LOAD_DECISION_STATES[len(_V1_LOAD_STATES):] == _PLUGIN_LOAD_ADDITIONS


def test_v11_throughput_additions_appended_in_order():
    assert (
        THROUGHPUT_DECISION_STATES[len(_V1_THROUGHPUT_STATES):]
        == _PLUGIN_THROUGHPUT_ADDITIONS
    )


@pytest.mark.parametrize("state", _PLUGIN_LOAD_ADDITIONS)
def test_new_load_state_is_settable_on_enum(state):
    """Construct an Enum with our extended states list and verify the
    new state can be set without raising. Uses an isolated registry so
    this test doesn't interfere with the module-level Prometheus
    registry."""
    registry = CollectorRegistry()
    gauge = Enum(
        "test_load_state",
        "test",
        states=LOAD_DECISION_STATES,
        registry=registry,
    )
    gauge.state(state)  # must not raise
    # Readback via collect() — the active state should be ours.
    samples = list(gauge.collect())[0].samples
    active = [s for s in samples if s.value == 1.0]
    assert len(active) == 1
    assert active[0].labels["test_load_state"] == state


@pytest.mark.parametrize("state", _PLUGIN_THROUGHPUT_ADDITIONS)
def test_new_throughput_state_is_settable_on_enum(state):
    registry = CollectorRegistry()
    gauge = Enum(
        "test_throughput_state",
        "test",
        states=THROUGHPUT_DECISION_STATES,
        registry=registry,
    )
    gauge.state(state)
    samples = list(gauge.collect())[0].samples
    active = [s for s in samples if s.value == 1.0]
    assert len(active) == 1
    assert active[0].labels["test_throughput_state"] == state


def test_load_state_list_has_no_duplicates():
    assert len(LOAD_DECISION_STATES) == len(set(LOAD_DECISION_STATES))


def test_throughput_state_list_has_no_duplicates():
    assert len(THROUGHPUT_DECISION_STATES) == len(
        set(THROUGHPUT_DECISION_STATES)
    )


def test_all_current_states_settable_end_to_end():
    """Defence-in-depth: each state in both lists must be settable on
    a freshly-constructed Enum gauge. Catches cases where a state name
    contains characters that Prometheus would silently accept at list
    construction but reject at state() call time (none today, but guards
    against future additions)."""
    for state in LOAD_DECISION_STATES:
        registry = CollectorRegistry()
        gauge = Enum(
            "test_all_load", "test", states=LOAD_DECISION_STATES, registry=registry
        )
        gauge.state(state)

    for state in THROUGHPUT_DECISION_STATES:
        registry = CollectorRegistry()
        gauge = Enum(
            "test_all_throughput",
            "test",
            states=THROUGHPUT_DECISION_STATES,
            registry=registry,
        )
        gauge.state(state)
