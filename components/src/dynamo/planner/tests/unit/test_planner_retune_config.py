# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The field classification of the Planner adapter's retune schema."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.reconfiguration import PREDICTOR_FIELDS, RUNTIME_FIELDS
from dynamo.planner.simulation.config import (
    PLANNER_PRESET_GROUPS,
    PLANNER_PUBLIC_TO_CONFIG,
    PLANNER_RETUNE_GROUPS,
    PLANNER_RETUNE_KNOBS,
    PLANNER_RETUNE_STARTUP_ONLY,
    PlannerRetuneConfig,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_every_retune_field_is_classified_and_reaches_a_runtime_field() -> None:
    # A new recommendation knob fails here until it is classified for retune.
    # Group knobs a running planner can change are searched through their
    # group; the others (load_min_observations) must be startup-only.
    runtime_group_knobs = {
        knob
        for group in PLANNER_RETUNE_GROUPS
        for knob in PLANNER_PRESET_GROUPS[group]
        if knob in RUNTIME_FIELDS
    }
    classes = [
        PLANNER_RETUNE_STARTUP_ONLY,
        set(PLANNER_RETUNE_KNOBS),
        set(PLANNER_RETUNE_GROUPS),
        runtime_group_knobs,
    ]
    fields = set(PlannerRetuneConfig.model_fields)
    assert set().union(*classes) == fields
    assert sum(len(names) for names in classes) == len(fields)
    # Retune reaches every runtime field except the SLA.
    reachable = (
        set(PLANNER_RETUNE_KNOBS.values())
        | {"load_scaling_down_sensitivity"}
        | PREDICTOR_FIELDS
    )
    assert reachable == RUNTIME_FIELDS - {"ttft_ms", "itl_ms"}
    assert all(
        PLANNER_PUBLIC_TO_CONFIG.get(public, public) == live
        for public, live in PLANNER_RETUNE_KNOBS.items()
    )
