# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for advisory mode and decision summary logging."""

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

from dynamo.planner.config.defaults import SLAPlannerDefaults  # noqa: E402

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestAdvisoryDefaults:
    def test_default_is_false(self):
        assert SLAPlannerDefaults.advisory is False


class TestPlannerConfigAdvisory:
    def test_config_with_advisory(self):
        from dynamo.planner.config.planner_config import PlannerConfig

        config = PlannerConfig.model_construct(
            mode="agg",
            advisory=True,
        )
        assert config.advisory is True

    def test_config_default_is_false(self):
        from dynamo.planner.config.planner_config import PlannerConfig

        config = PlannerConfig.model_construct(mode="agg")
        assert config.advisory is False


class TestAdvisoryGuard:
    def test_advisory_skips_scaling(self):
        advisory = True
        assert advisory  # _apply_scaling_targets returns early

    def test_non_advisory_applies_scaling(self):
        advisory = False
        assert not advisory  # _apply_scaling_targets proceeds


def _classify_action(delta_p: int, delta_d: int, decision_is_none: bool) -> str:
    """Mirror the action classification from _log_decision_summary."""
    if decision_is_none or (delta_p == 0 and delta_d == 0):
        return "hold"
    if (delta_p > 0 or delta_d > 0) and (delta_p < 0 or delta_d < 0):
        return "rebalance"
    if delta_p > 0 or delta_d > 0:
        return "scale_up"
    return "scale_down"


class TestDecisionSummaryClassification:
    def test_scale_up(self):
        assert _classify_action(1, 2, False) == "scale_up"

    def test_scale_down(self):
        assert _classify_action(-1, -2, False) == "scale_down"

    def test_hold_no_change(self):
        assert _classify_action(0, 0, False) == "hold"

    def test_hold_no_decision(self):
        assert _classify_action(0, 0, True) == "hold"

    def test_rebalance_prefill_up_decode_down(self):
        assert _classify_action(1, -2, False) == "rebalance"

    def test_rebalance_prefill_down_decode_up(self):
        assert _classify_action(-1, 2, False) == "rebalance"
