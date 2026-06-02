# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SchedulingConfig + PlannerConfig.scheduling."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dynamo.planner.config.planner_config import PlannerConfig, SchedulingConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# SchedulingConfig defaults
# ---------------------------------------------------------------------------


def test_scheduling_defaults_opt_out_of_orchestrator():
    """Feature flag default MUST be False — upgrading existing
    deployments shouldn't flip behavior until operators opt in."""
    s = SchedulingConfig()
    assert s.use_orchestrator is False


def test_scheduling_default_timeouts_match_spec():
    s = SchedulingConfig()
    assert s.tick_max_duration_seconds == 30.0


def test_scheduling_rejects_non_positive_tick_deadline():
    with pytest.raises(ValidationError):
        SchedulingConfig(tick_max_duration_seconds=0)


# ---------------------------------------------------------------------------
# scale_interval_seconds (scale_interval cadence model surface)
# ---------------------------------------------------------------------------


def test_scale_interval_default_matches_load_cadence():
    """Default 5.0s matches the default load_adjustment_interval_seconds
    so an orchestrator-path deployment with default config sees
    pipeline ticks at the same cadence the load cadence used to drive
    (under the legacy PSM-mirror model). Behaviour-equivalent default."""
    s = SchedulingConfig()
    assert s.scale_interval_seconds == 5.0


def test_scale_interval_rejects_zero_or_negative():
    """Pipeline cannot fire at zero or negative cadence. Pydantic gt=0
    catches at config-validation time."""
    with pytest.raises(ValidationError):
        SchedulingConfig(scale_interval_seconds=0)
    with pytest.raises(ValidationError):
        SchedulingConfig(scale_interval_seconds=-1.0)


def test_scale_interval_accepts_explicit_value():
    """Operators can override the default — e.g., set a 1s cadence for
    high-frequency scenarios. Lock that the field accepts positive
    floats."""
    s = SchedulingConfig(scale_interval_seconds=1.0)
    assert s.scale_interval_seconds == 1.0


# ---------------------------------------------------------------------------
# PlannerConfig integration — backwards compat
# ---------------------------------------------------------------------------


def test_planner_config_default_has_scheduling_subtree():
    pc = PlannerConfig()
    assert isinstance(pc.scheduling, SchedulingConfig)
    assert pc.scheduling.use_orchestrator is False


def test_planner_config_without_scheduling_section_loads_unchanged():
    """Existing yaml configs (pre-PR-7) don't have a scheduling section.
    They must continue to load with default SchedulingConfig."""
    # A minimal config — no scheduling key.
    raw = yaml.safe_dump(
        {
            "mode": "disagg",
            "environment": "kubernetes",
            "enable_throughput_scaling": True,
        }
    )
    loaded = yaml.safe_load(raw)
    pc = PlannerConfig.model_validate(loaded)
    assert pc.scheduling.use_orchestrator is False
    assert pc.scheduling.tick_max_duration_seconds == 30.0


def test_planner_config_with_scheduling_override_parses():
    pc = PlannerConfig.model_validate(
        {
            "mode": "disagg",
            "environment": "kubernetes",
            "enable_throughput_scaling": True,
            "scheduling": {
                "use_orchestrator": True,
                "tick_max_duration_seconds": 60.0,
            },
        }
    )
    assert pc.scheduling.use_orchestrator is True
    assert pc.scheduling.tick_max_duration_seconds == 60.0


def test_planner_config_yaml_round_trip_preserves_scheduling():
    pc = PlannerConfig.model_validate(
        {
            "mode": "disagg",
            "environment": "kubernetes",
            "enable_throughput_scaling": True,
            "scheduling": {"use_orchestrator": True},
        }
    )
    # mode="json" projects enums → strings so yaml.safe_dump is happy.
    dumped = yaml.safe_dump(pc.model_dump(mode="json"))
    reloaded = yaml.safe_load(dumped)
    pc2 = PlannerConfig.model_validate(reloaded)
    assert pc2.scheduling.use_orchestrator is True


def test_planner_config_partial_scheduling_override_keeps_other_defaults():
    pc = PlannerConfig.model_validate(
        {
            "mode": "disagg",
            "environment": "kubernetes",
            "enable_throughput_scaling": True,
            "scheduling": {"use_orchestrator": True},  # only flip the flag
        }
    )
    assert pc.scheduling.use_orchestrator is True
    # Other fields take defaults.
    assert pc.scheduling.tick_max_duration_seconds == 30.0
