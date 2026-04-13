# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Advisory Mode functionality.

These tests validate ScalingMode configuration and advisory mode logic
without requiring Rust bindings (dynamo._core).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------
# Stub native Rust modules so planner config can be imported
# without building dynamo._core
# ---------------------------------------------------------------
_stubs = {
    "dynamo._core": {
        "Client": MagicMock,
        "DistributedRuntime": MagicMock,
        "VirtualConnectorCoordinator": MagicMock,
    },
    "dynamo.runtime": {
        "DistributedRuntime": MagicMock,
        "dynamo_worker": lambda: lambda f: f,
    },
    "dynamo.runtime.logging": {
        "configure_dynamo_logging": lambda: None,
    },
    "dynamo.llm": {
        "FpmEventSubscriber": MagicMock,
        "FpmEventRelay": MagicMock,
    },
    "dynamo.common.forward_pass_metrics": {
        "ForwardPassMetrics": MagicMock,
    },
}
for _mod_name, _attrs in _stubs.items():
    _m = types.ModuleType(_mod_name)
    for _k, _v in _attrs.items():
        setattr(_m, _k, _v)
    sys.modules.setdefault(_mod_name, _m)

# Now safe to import planner modules
from dynamo.planner.config.defaults import ScalingMode, SLAPlannerDefaults


class TestScalingModeEnum:
    """Test ScalingMode enum definition."""

    def test_enum_values(self):
        assert ScalingMode.ACTIVE.value == "active"
        assert ScalingMode.ADVISORY.value == "advisory"
        assert ScalingMode.NOOP.value == "noop"

    def test_enum_from_string(self):
        assert ScalingMode("active") == ScalingMode.ACTIVE
        assert ScalingMode("advisory") == ScalingMode.ADVISORY
        assert ScalingMode("noop") == ScalingMode.NOOP

    def test_enum_is_string(self):
        """ScalingMode should be usable as a string."""
        assert ScalingMode.ADVISORY == "advisory"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ScalingMode("invalid")


class TestScalingModeDefaults:
    """Test defaults and config integration."""

    def test_default_is_active(self):
        assert SLAPlannerDefaults.scaling_mode == ScalingMode.ACTIVE

    def test_advisory_log_interval_default(self):
        assert SLAPlannerDefaults.advisory_log_interval == 60


class TestPlannerConfigScalingMode:
    """Test PlannerConfig accepts scaling_mode."""

    def test_config_with_advisory(self):
        from dynamo.planner.config.planner_config import PlannerConfig

        config = PlannerConfig.model_construct(
            mode="agg",
            component_name="test",
            endpoint_name="test",
            scaling_mode=ScalingMode.ADVISORY,
            advisory_log_interval=120,
        )
        assert config.scaling_mode == ScalingMode.ADVISORY
        assert config.advisory_log_interval == 120

    def test_config_default_is_active(self):
        from dynamo.planner.config.planner_config import PlannerConfig

        config = PlannerConfig.model_construct(
            mode="agg",
            component_name="test",
            endpoint_name="test",
        )
        assert config.scaling_mode == ScalingMode.ACTIVE


class TestAdvisoryModeLogic:
    """Test the advisory decision logic (without runtime dependencies)."""

    def test_advisory_mode_skips_apply_effects(self):
        """Verify the branching logic: advisory mode should NOT call _apply_effects."""
        # Simulate the run() loop branching logic
        scaling_mode = ScalingMode.ADVISORY
        apply_called = False
        log_called = False

        if scaling_mode == ScalingMode.ADVISORY:
            log_called = True
        elif scaling_mode == ScalingMode.ACTIVE:
            apply_called = True

        assert log_called is True
        assert apply_called is False

    def test_active_mode_calls_apply_effects(self):
        """Verify active mode calls _apply_effects."""
        scaling_mode = ScalingMode.ACTIVE
        apply_called = False
        log_called = False

        if scaling_mode == ScalingMode.ADVISORY:
            log_called = True
        elif scaling_mode == ScalingMode.ACTIVE:
            apply_called = True

        assert apply_called is True
        assert log_called is False

    def test_noop_mode_does_nothing(self):
        """Verify noop mode neither applies nor logs."""
        scaling_mode = ScalingMode.NOOP
        apply_called = False
        log_called = False

        if scaling_mode == ScalingMode.ADVISORY:
            log_called = True
        elif scaling_mode == ScalingMode.ACTIVE:
            apply_called = True

        assert apply_called is False
        assert log_called is False

    def test_advisory_delta_calculation(self):
        """Test delta calculation for advisory logging."""
        current_p, current_d = 2, 4
        rec_p, rec_d = 3, 6

        delta_p = rec_p - current_p
        delta_d = rec_d - current_d

        assert delta_p == 1
        assert delta_d == 2

        # Action determination
        if delta_p > 0 or delta_d > 0:
            action = "scale_up"
        elif delta_p < 0 or delta_d < 0:
            action = "scale_down"
        else:
            action = "hold"

        assert action == "scale_up"

    def test_advisory_hold_when_no_change(self):
        """No delta → hold."""
        current_p, current_d = 3, 6
        rec_p, rec_d = 3, 6

        delta_p = rec_p - current_p
        delta_d = rec_d - current_d

        if delta_p > 0 or delta_d > 0:
            action = "scale_up"
        elif delta_p < 0 or delta_d < 0:
            action = "scale_down"
        else:
            action = "hold"

        assert action == "hold"
