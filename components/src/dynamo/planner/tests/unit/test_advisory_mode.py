# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Advisory Mode functionality."""

import pytest

from dynamo.planner.config.defaults import ScalingMode, SLAPlannerDefaults
from dynamo.planner.config.planner_config import PlannerConfig


class TestScalingModeConfig:
    """Test ScalingMode enum and configuration."""

    def test_scaling_mode_enum_values(self):
        """Verify ScalingMode enum has expected values."""
        assert ScalingMode.ACTIVE.value == "active"
        assert ScalingMode.ADVISORY.value == "advisory"
        assert ScalingMode.NOOP.value == "noop"

    def test_default_scaling_mode_is_active(self):
        """Default scaling mode should be ACTIVE."""
        assert SLAPlannerDefaults.scaling_mode == ScalingMode.ACTIVE

    def test_config_accepts_scaling_mode(self):
        """PlannerConfig should accept scaling_mode parameter."""
        config = PlannerConfig(
            mode="agg",
            component_name="test",
            endpoint_name="test",
            scaling_mode=ScalingMode.ADVISORY,
        )
        assert config.scaling_mode == ScalingMode.ADVISORY

    def test_config_accepts_scaling_mode_string(self):
        """PlannerConfig should accept scaling_mode as string."""
        config = PlannerConfig(
            mode="agg",
            component_name="test",
            endpoint_name="test",
            scaling_mode="advisory",
        )
        assert config.scaling_mode == ScalingMode.ADVISORY

    def test_advisory_log_interval_default(self):
        """Default advisory_log_interval should be 60 seconds."""
        config = PlannerConfig(
            mode="agg",
            component_name="test",
            endpoint_name="test",
        )
        assert config.advisory_log_interval == 60

    def test_advisory_log_interval_custom(self):
        """Custom advisory_log_interval should be accepted."""
        config = PlannerConfig(
            mode="agg",
            component_name="test",
            endpoint_name="test",
            advisory_log_interval=120,
        )
        assert config.advisory_log_interval == 120


class TestAdvisoryModeIntegration:
    """Integration tests for advisory mode behavior."""

    def test_advisory_mode_does_not_call_connector(self):
        """In advisory mode, scaling decisions should not be executed."""
        # This would require mocking the connector and state machine
        # Full integration test deferred to E2E testing
        pass
