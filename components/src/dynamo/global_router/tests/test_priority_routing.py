#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for priority-based pool routing in the global router."""

import json
from pathlib import Path

import pytest

from dynamo.global_router.pool_selection import (
    DecodePoolSelectionStrategy,
    GlobalRouterConfig,
    PrefillPoolSelectionStrategy,
    PriorityPoolOverride,
    _apply_priority_overrides,
    get_priority_retry_order,
    load_config,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]

# --- _apply_priority_overrides tests ---


class TestApplyPriorityOverrides:
    def test_no_priority_returns_base(self):
        rules = [PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=1)]
        assert _apply_priority_overrides(0, None, rules) == 0

    def test_no_rules_returns_base(self):
        assert _apply_priority_overrides(0, 5, []) == 0

    def test_match_returns_target(self):
        rules = [PriorityPoolOverride(min_priority=5, max_priority=15, target_pool=2)]
        assert _apply_priority_overrides(0, 10, rules) == 2

    def test_no_match_returns_base(self):
        rules = [PriorityPoolOverride(min_priority=5, max_priority=15, target_pool=2)]
        assert _apply_priority_overrides(0, 20, rules) == 0

    def test_first_match_wins(self):
        rules = [
            PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=1),
            PriorityPoolOverride(min_priority=5, max_priority=20, target_pool=2),
        ]
        # priority=7 matches both rules; first should win
        assert _apply_priority_overrides(0, 7, rules) == 1

    def test_boundary_inclusive(self):
        rules = [PriorityPoolOverride(min_priority=5, max_priority=10, target_pool=1)]
        assert _apply_priority_overrides(0, 5, rules) == 1
        assert _apply_priority_overrides(0, 10, rules) == 1
        assert _apply_priority_overrides(0, 4, rules) == 0
        assert _apply_priority_overrides(0, 11, rules) == 0

    def test_empty_priority_and_empty_rules(self):
        assert _apply_priority_overrides(3, None, []) == 3


# --- Priority retry order tests ---


class TestPriorityRetryOrder:
    def test_disabled_only_returns_selected_pool(self):
        assert get_priority_retry_order(2, [0, 1, 2], False) == [2]

    def test_default_priorities_retry_from_slow_to_fast(self):
        assert get_priority_retry_order(2, [0, 1, 2], True) == [2, 1, 0]

    def test_fastest_pool_has_no_faster_retry(self):
        assert get_priority_retry_order(0, [0, 1, 2], True) == [0]

    def test_custom_priorities_retry_by_next_faster_priority(self):
        # Pool 0 is slowest, pool 2 is middle, pool 1 is fastest.
        assert get_priority_retry_order(0, [10, 0, 5], True) == [0, 2, 1]

    def test_same_priority_pools_preserve_pool_order(self):
        assert get_priority_retry_order(3, [0, 1, 1, 2], True) == [3, 1, 2, 0]


# --- PrefillPoolSelectionStrategy with priority ---


def _make_prefill_strategy(
    num_pools=2, priority_overrides=None
) -> PrefillPoolSelectionStrategy:
    return PrefillPoolSelectionStrategy(
        ttft_min_ms=10,
        ttft_max_ms=3000,
        ttft_resolution=2,
        isl_min=0,
        isl_max=32000,
        isl_resolution=2,
        prefill_pool_mapping=[[0, 1], [0, 1]],
        priority_overrides=priority_overrides or [],
    )


class TestPrefillSelectPoolWithPriority:
    def test_priority_override_takes_precedence(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=1)
            ]
        )
        # Grid would select pool 0 (low ISL, low TTFT), but priority overrides
        result = strategy.select_pool(isl=100, ttft_target_ms=50, priority=50)
        assert result == 1

    def test_no_priority_uses_grid(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=1)
            ]
        )
        result = strategy.select_pool(isl=100, ttft_target_ms=50)
        assert result == 0  # grid result, no priority

    def test_unmatched_priority_uses_grid(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=1)
            ]
        )
        result = strategy.select_pool(isl=100, ttft_target_ms=50, priority=5)
        assert result == 0  # priority=5 doesn't match [10, 100]

    def test_no_overrides_backward_compatible(self):
        strategy = _make_prefill_strategy()
        result = strategy.select_pool(isl=100, ttft_target_ms=50, priority=50)
        assert result == 0  # no overrides configured, grid result


# --- DecodePoolSelectionStrategy with priority ---


def _make_decode_strategy(
    num_pools=2, priority_overrides=None
) -> DecodePoolSelectionStrategy:
    return DecodePoolSelectionStrategy(
        itl_min_ms=10,
        itl_max_ms=500,
        itl_resolution=2,
        context_length_min=0,
        context_length_max=32000,
        context_length_resolution=2,
        decode_pool_mapping=[[0, 1], [0, 1]],
        priority_overrides=priority_overrides or [],
    )


class TestDecodeSelectPoolWithPriority:
    def test_priority_override_takes_precedence(self):
        strategy = _make_decode_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=1)
            ]
        )
        result = strategy.select_pool(context_length=100, itl_target_ms=20, priority=50)
        assert result == 1

    def test_no_priority_uses_grid(self):
        strategy = _make_decode_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=1)
            ]
        )
        result = strategy.select_pool(context_length=100, itl_target_ms=20)
        assert result == 0


# --- Config loading tests ---


def _write_config(tmp_dir: Path, config_data: dict) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _base_config(**overrides) -> dict:
    config = {
        "num_prefill_pools": 2,
        "num_decode_pools": 2,
        "prefill_pool_dynamo_namespaces": ["ns-prefill-0", "ns-prefill-1"],
        "decode_pool_dynamo_namespaces": ["ns-decode-0", "ns-decode-1"],
        "prefill_pool_selection_strategy": {
            "isl_min": 0,
            "isl_max": 32000,
            "isl_resolution": 2,
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 2,
            "prefill_pool_mapping": [[0, 1], [0, 1]],
        },
        "decode_pool_selection_strategy": {
            "context_length_min": 0,
            "context_length_max": 32000,
            "context_length_resolution": 2,
            "itl_min": 10,
            "itl_max": 500,
            "itl_resolution": 2,
            "decode_pool_mapping": [[0, 1], [0, 1]],
        },
    }
    config.update(overrides)
    return config


class TestLoadConfigWithPriorityOverrides:
    def test_with_priority_overrides(self, tmp_path):
        config_data = _base_config()
        config_data["prefill_pool_selection_strategy"]["priority_overrides"] = [
            {"min_priority": 10, "max_priority": 100, "target_pool": 1}
        ]
        config_data["decode_pool_selection_strategy"]["priority_overrides"] = [
            {"min_priority": 5, "max_priority": 50, "target_pool": 0}
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert len(config.prefill_pool_selection_strategy.priority_overrides) == 1
        override = config.prefill_pool_selection_strategy.priority_overrides[0]
        assert override.min_priority == 10
        assert override.max_priority == 100
        assert override.target_pool == 1

        assert len(config.decode_pool_selection_strategy.priority_overrides) == 1
        override = config.decode_pool_selection_strategy.priority_overrides[0]
        assert override.min_priority == 5
        assert override.max_priority == 50
        assert override.target_pool == 0

    def test_without_priority_overrides(self, tmp_path):
        config_data = _base_config()
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert config.prefill_pool_selection_strategy.priority_overrides == []
        assert config.decode_pool_selection_strategy.priority_overrides == []


class TestLoadConfigWithPriorityRetry:
    def test_defaults_pool_priorities_from_pool_order(self, tmp_path):
        config_data = _base_config()
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert config.enable_priority_retry is False
        assert config.prefill_pool_priorities == [0, 1]
        assert config.decode_pool_priorities == [0, 1]

    def test_loads_explicit_pool_priorities_and_retry_flag(self, tmp_path):
        config_data = _base_config(
            enable_priority_retry=True,
            prefill_pool_priorities=[10, 0],
            decode_pool_priorities=[5, 1],
        )
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert config.enable_priority_retry is True
        assert config.prefill_pool_priorities == [10, 0]
        assert config.decode_pool_priorities == [5, 1]


# --- Validation tests ---


class TestValidatePriorityOverrides:
    def test_invalid_prefill_target_pool(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=5)
            ]
        )
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=strategy,
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        with pytest.raises(ValueError, match="invalid target_pool"):
            config.validate()

    def test_invalid_decode_target_pool(self):
        strategy = _make_decode_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=3)
            ]
        )
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=_make_prefill_strategy(),
            decode_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="invalid target_pool"):
            config.validate()

    def test_min_greater_than_max(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=20, max_priority=5, target_pool=1)
            ]
        )
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=strategy,
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        with pytest.raises(ValueError, match="min_priority"):
            config.validate()

    def test_valid_overrides_pass(self):
        strategy = _make_prefill_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=1)
            ]
        )
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=strategy,
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        config.validate()  # should not raise

    def test_default_mode_is_disagg(self, tmp_path):
        """Config without explicit mode defaults to disagg."""
        config_data = _base_config()
        # No "mode" key
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)
        assert config.mode == "disagg"


class TestValidatePriorityRetry:
    def test_invalid_prefill_pool_priorities_length(self):
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_priorities=[0],
            prefill_pool_selection_strategy=_make_prefill_strategy(),
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        with pytest.raises(ValueError, match="prefill_pool_priorities length"):
            config.validate()

    def test_invalid_decode_pool_priority_type(self):
        config = GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            decode_pool_priorities=[0, "1"],
            prefill_pool_selection_strategy=_make_prefill_strategy(),
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        with pytest.raises(ValueError, match="decode_pool_priorities\\[1\\]"):
            config.validate()

    def test_enable_priority_retry_must_be_bool(self):
        config = GlobalRouterConfig(
            mode="disagg",
            enable_priority_retry="yes",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=_make_prefill_strategy(),
            decode_pool_selection_strategy=_make_decode_strategy(),
        )
        with pytest.raises(ValueError, match="enable_priority_retry"):
            config.validate()
