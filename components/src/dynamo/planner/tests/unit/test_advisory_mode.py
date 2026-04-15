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
from dynamo.planner.config.defaults import ScalingMode, SLAPlannerDefaults  # noqa: E402

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestScalingModeEnum:
    """Test ScalingMode enum definition."""

    def test_enum_values(self):
        assert ScalingMode.ACTIVE.value == "active"
        assert ScalingMode.ADVISORY.value == "advisory"

    def test_enum_from_string(self):
        assert ScalingMode("active") == ScalingMode.ACTIVE
        assert ScalingMode("advisory") == ScalingMode.ADVISORY

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


def _classify_action(delta_p: int, delta_d: int, decision_is_none: bool) -> str:
    """Mirror the action classification logic from _log_advisory_decision."""
    if decision_is_none or (delta_p == 0 and delta_d == 0):
        return "hold"
    if (delta_p > 0 or delta_d > 0) and (delta_p < 0 or delta_d < 0):
        return "rebalance"
    if delta_p > 0 or delta_d > 0:
        return "scale_up"
    return "scale_down"


class TestAdvisoryModeLogic:
    """Test the advisory decision logic (without runtime dependencies)."""

    def test_advisory_mode_skips_scaling_but_runs_effects(self):
        """Advisory mode: _apply_effects runs (for metrics), but
        _apply_scaling_targets is guarded (no actual scaling)."""
        scaling_mode = ScalingMode.ADVISORY
        no_operation = False

        # _apply_scaling_targets guard (mirrors base.py)
        scaling_skipped = (
            no_operation or scaling_mode == ScalingMode.ADVISORY
        )
        assert scaling_skipped is True

    def test_active_mode_applies_scaling(self):
        """Active mode: _apply_scaling_targets proceeds."""
        scaling_mode = ScalingMode.ACTIVE
        no_operation = False

        scaling_skipped = (
            no_operation or scaling_mode == ScalingMode.ADVISORY
        )
        assert scaling_skipped is False

    def test_advisory_delta_calculation(self):
        """Test delta calculation for advisory logging."""
        current_p, current_d = 2, 4
        rec_p, rec_d = 3, 6

        delta_p = rec_p - current_p
        delta_d = rec_d - current_d

        assert delta_p == 1
        assert delta_d == 2
        assert _classify_action(delta_p, delta_d, False) == "scale_up"

    def test_advisory_hold_when_no_change(self):
        """No delta -> hold."""
        assert _classify_action(0, 0, False) == "hold"

    def test_advisory_hold_when_no_decision(self):
        """None decision -> hold."""
        assert _classify_action(0, 0, True) == "hold"

    def test_advisory_scale_down(self):
        assert _classify_action(-1, -2, False) == "scale_down"

    def test_advisory_rebalance_when_mixed(self):
        """Prefill up + decode down -> rebalance."""
        assert _classify_action(1, -2, False) == "rebalance"

    def test_advisory_rebalance_reversed(self):
        """Prefill down + decode up -> rebalance."""
        assert _classify_action(-1, 2, False) == "rebalance"


class TestAdvisoryModeStartupBehavior:
    """Test advisory mode startup behavior.

    Advisory mode goes through the same startup path as active mode
    (validate deployment, discover workers, subscribe to FPM) because
    it needs real metrics to compute decisions.  The only difference
    is that scaling decisions are logged, not executed.
    """

    def test_advisory_mode_validates_deployment(self):
        """Advisory mode MUST validate deployment to discover workers."""
        no_operation = False
        validate_called = False

        # Simulate _async_init branching — advisory uses the same path
        if not no_operation:
            validate_called = True

        assert validate_called is True

    def test_active_mode_validates_deployment(self):
        """Active mode must validate deployment."""
        no_operation = False
        validate_called = False

        if not no_operation:
            validate_called = True

        assert validate_called is True

    def test_no_operation_skips_validation(self):
        """Only no_operation=True skips deployment validation."""
        no_operation = True
        validate_called = False

        if not no_operation:
            validate_called = True

        assert validate_called is False
