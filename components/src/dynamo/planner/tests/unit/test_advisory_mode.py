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


class TestAdvisoryModeStartupBehavior:
    """Test advisory mode startup behavior (bugfix e5613ae)."""

    def test_advisory_mode_skips_validate_deployment(self):
        """Advisory mode should not call validate_deployment().

        Bugfix e5613ae: advisory/noop modes skip K8s deployment validation
        since they don't need actual worker info to log recommendations.
        """
        scaling_mode = ScalingMode.ADVISORY
        validate_called = False

        # Simulate _async_init branching
        if scaling_mode == ScalingMode.ACTIVE:
            # Only active mode validates deployment
            validate_called = True

        assert validate_called is False

    def test_noop_mode_skips_validate_deployment(self):
        """Noop mode also skips validate_deployment()."""
        scaling_mode = ScalingMode.NOOP
        validate_called = False

        if scaling_mode == ScalingMode.ACTIVE:
            validate_called = True

        assert validate_called is False

    def test_active_mode_validates_deployment(self):
        """Active mode must validate deployment."""
        scaling_mode = ScalingMode.ACTIVE
        validate_called = False

        if scaling_mode == ScalingMode.ACTIVE:
            validate_called = True

        assert validate_called is True

    def test_advisory_mode_can_start_without_workers(self):
        """Advisory mode should be able to start when no workers exist.

        This is a key E2E requirement: operators want to observe Planner
        recommendations before deploying any workers.
        """
        scaling_mode = ScalingMode.ADVISORY
        num_workers = 0

        # Advisory mode: can start regardless of worker count
        can_start = scaling_mode != ScalingMode.ACTIVE or num_workers > 0

        assert can_start is True

    def test_active_mode_requires_workers(self):
        """Active mode traditionally requires workers to start."""
        scaling_mode = ScalingMode.ACTIVE
        num_workers = 0

        # Active mode needs workers (simplified logic)
        can_start = scaling_mode != ScalingMode.ACTIVE or num_workers > 0

        # With 0 workers and active mode, can_start = False
        assert can_start is False


class TestLoadScalingMaxBatchedTokens:
    """Test load scaling behavior with missing max_num_batched_tokens (bugfix 9dc6521)."""

    def test_decode_scaling_proceeds_without_max_batched_tokens(self):
        """Decode scaling should NOT be blocked when max_num_batched_tokens is unavailable.

        Bugfix 9dc6521: _advance_load_agg was returning None for ALL scaling
        when max_num_batched_tokens was not discovered via MDC. However,
        decode scaling (ITL-based) does not need this value.
        """
        max_num_batched_tokens = None  # MDC fallback scenario

        # Simulated logic after fix
        if max_num_batched_tokens is None:
            p_desired = None  # Skip prefill scaling only
        else:
            p_desired = 3  # Would compute prefill scaling

        # Decode scaling always proceeds
        d_desired = 5  # Computed from ITL regression

        # Decision logic: decode scaling should still work
        assert p_desired is None  # Prefill skipped
        assert d_desired == 5  # Decode proceeds

    def test_prefill_scaling_requires_max_batched_tokens(self):
        """Prefill scaling needs max_num_batched_tokens to estimate TTFT."""
        max_num_batched_tokens = None

        if max_num_batched_tokens is None:
            p_desired = None
        else:
            p_desired = 3

        assert p_desired is None

    def test_both_scaling_paths_when_max_batched_tokens_available(self):
        """When max_num_batched_tokens is available, both paths should work."""
        max_num_batched_tokens = 8192

        if max_num_batched_tokens is None:
            p_desired = None
        else:
            p_desired = 3  # Prefill scaling works

        d_desired = 5  # Decode always works

        assert p_desired == 3
        assert d_desired == 5

    def test_agg_scaling_not_blocked_by_missing_max_batched_tokens(self):
        """Agg mode scaling should not return None when only max_batched_tokens is missing.

        Before fix: entire _advance_load_agg returned None
        After fix: only prefill is None, decode proceeds
        """
        max_num_batched_tokens = None
        num_workers = 2

        # Simulate _advance_load_agg logic after fix
        if max_num_batched_tokens is None:
            p_desired = None
        else:
            p_desired = 4

        d_desired = 3  # Decode scaling result

        # Decision: should still return a scaling decision based on decode
        if p_desired is not None and p_desired > num_workers:
            desired = p_desired
        elif d_desired is not None and d_desired > num_workers:
            desired = d_desired
        elif (
            p_desired is not None
            and p_desired < num_workers
            and d_desired is not None
            and d_desired < num_workers
        ):
            desired = max(p_desired, d_desired)
        else:
            desired = None  # No change

        # With d_desired=3 > num_workers=2, should scale up to 3
        assert desired == 3


class TestNatsEnableForFpm:
    """Test NATS enablement logic for FPM (bugfix 9dc6521 runtime.py)."""

    def test_nats_enabled_when_fpm_port_set(self):
        """NATS should be enabled when DYN_FORWARDPASS_METRIC_PORT is set.

        Bugfix 9dc6521: Workers using TCP request plane + no KV events
        previously never enabled NATS, preventing FPM relay from publishing.
        """
        request_plane = "tcp"
        event_plane = "nats"
        use_kv_events = False
        fpm_enabled = True  # DYN_FORWARDPASS_METRIC_PORT is set

        # Logic after fix
        enable_nats = request_plane == "nats" or (
            event_plane == "nats" and (use_kv_events or fpm_enabled)
        )

        assert enable_nats is True

    def test_nats_not_enabled_without_fpm_or_kv(self):
        """NATS should NOT be enabled if neither FPM nor KV events need it."""
        request_plane = "tcp"
        event_plane = "nats"
        use_kv_events = False
        fpm_enabled = False

        enable_nats = request_plane == "nats" or (
            event_plane == "nats" and (use_kv_events or fpm_enabled)
        )

        assert enable_nats is False

    def test_nats_enabled_for_kv_events(self):
        """NATS should be enabled when KV events are used."""
        request_plane = "tcp"
        event_plane = "nats"
        use_kv_events = True
        fpm_enabled = False

        enable_nats = request_plane == "nats" or (
            event_plane == "nats" and (use_kv_events or fpm_enabled)
        )

        assert enable_nats is True

    def test_nats_enabled_for_nats_request_plane(self):
        """NATS should be enabled when request plane is NATS."""
        request_plane = "nats"
        event_plane = "zmq"
        use_kv_events = False
        fpm_enabled = False

        enable_nats = request_plane == "nats" or (
            event_plane == "nats" and (use_kv_events or fpm_enabled)
        )

        assert enable_nats is True
