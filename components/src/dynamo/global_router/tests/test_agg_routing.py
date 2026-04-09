#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for aggregated (agg) mode pool routing in the global router."""

import json
from pathlib import Path

import pytest

from dynamo.global_router.pool_selection import (
    AggPoolSelectionStrategy,
    GlobalRouterConfig,
    PriorityPoolOverride,
    load_config,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


# --- Helpers ---


def _make_agg_strategy(
    num_pools=2, priority_overrides=None
) -> AggPoolSelectionStrategy:
    return AggPoolSelectionStrategy(
        ttft_min=10,
        ttft_max=3000,
        ttft_resolution=2,
        itl_min=5,
        itl_max=200,
        itl_resolution=2,
        agg_pool_mapping=[[0, 1], [1, 1]],
        priority_overrides=priority_overrides or [],
    )


def _write_config(tmp_dir: Path, config_data: dict) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _agg_base_config(**overrides) -> dict:
    config = {
        "mode": "agg",
        "num_agg_pools": 2,
        "agg_pool_dynamo_namespaces": ["ns-agg-0", "ns-agg-1"],
        "agg_pool_selection_strategy": {
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 2,
            "itl_min": 5,
            "itl_max": 200,
            "itl_resolution": 2,
            "agg_pool_mapping": [[0, 1], [1, 1]],
        },
    }
    config.update(overrides)
    return config


# --- AggPoolSelectionStrategy tests ---


class TestAggPoolSelection:
    """Tests for the TTFT x ITL grid selection.

    Default strategy mapping [[0, 1], [1, 1]] with:
    - TTFT: [10, 3000], resolution 2 -> step=1495, boundary at 1505
    - ITL:  [5, 200],   resolution 2 -> step=97.5, boundary at 102.5

    mapping[ttft_idx][itl_idx]:
    - [0][0] = 0: tight TTFT + tight ITL  -> pool 0
    - [0][1] = 1: tight TTFT + relaxed ITL -> pool 1
    - [1][0] = 1: relaxed TTFT + tight ITL -> pool 1
    - [1][1] = 1: relaxed TTFT + relaxed ITL -> pool 1
    """

    def test_tight_ttft_tight_itl(self):
        strategy = _make_agg_strategy()
        # ttft_idx=0, itl_idx=0 -> pool 0
        result = strategy.select_pool(ttft_target=100, itl_target=10)
        assert result == 0

    def test_tight_ttft_relaxed_itl(self):
        strategy = _make_agg_strategy()
        # ttft_idx=0, itl_idx=1 -> pool 1
        result = strategy.select_pool(ttft_target=100, itl_target=150)
        assert result == 1

    def test_relaxed_ttft_tight_itl(self):
        strategy = _make_agg_strategy()
        # ttft_idx=1, itl_idx=0 -> pool 1
        result = strategy.select_pool(ttft_target=2000, itl_target=10)
        assert result == 1

    def test_relaxed_ttft_relaxed_itl(self):
        strategy = _make_agg_strategy()
        # ttft_idx=1, itl_idx=1 -> pool 1
        result = strategy.select_pool(ttft_target=2000, itl_target=150)
        assert result == 1

    def test_default_ttft_uses_midpoint(self):
        strategy = _make_agg_strategy()
        # ttft_target=None -> midpoint=(10+3000)/2=1505 -> ttft_idx=1
        # itl_target=10 -> itl_idx=0
        # [1][0] = 1
        result = strategy.select_pool(itl_target=10)
        assert result == 1

    def test_default_itl_uses_midpoint(self):
        strategy = _make_agg_strategy()
        # ttft_target=100 -> ttft_idx=0
        # itl_target=None -> midpoint=(5+200)/2=102.5 -> itl_idx=1
        # [0][1] = 1
        result = strategy.select_pool(ttft_target=100)
        assert result == 1

    def test_both_defaults_use_midpoints(self):
        strategy = _make_agg_strategy()
        # ttft midpoint=1505 -> ttft_idx=1, itl midpoint=102.5 -> itl_idx=1
        # [1][1] = 1
        result = strategy.select_pool()
        assert result == 1

    def test_clamping_below_min(self):
        strategy = _make_agg_strategy()
        # Both below min -> both clamp to idx 0
        result = strategy.select_pool(ttft_target=0, itl_target=0)
        assert result == 0

    def test_clamping_above_max(self):
        strategy = _make_agg_strategy()
        # Both above max -> both clamp to max idx
        result = strategy.select_pool(ttft_target=10000, itl_target=1000)
        assert result == 1

    def test_priority_override_takes_precedence(self):
        strategy = _make_agg_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=0)
            ]
        )
        # Grid: relaxed TTFT + relaxed ITL -> pool 1, but priority overrides to 0
        result = strategy.select_pool(ttft_target=2000, itl_target=150, priority=50)
        assert result == 0

    def test_no_priority_uses_grid(self):
        strategy = _make_agg_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=0)
            ]
        )
        result = strategy.select_pool(ttft_target=2000, itl_target=150)
        assert result == 1  # grid result, no priority

    def test_unmatched_priority_uses_grid(self):
        strategy = _make_agg_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=0)
            ]
        )
        result = strategy.select_pool(ttft_target=2000, itl_target=150, priority=5)
        assert result == 1  # priority=5 doesn't match [10, 100]

    def test_no_overrides_backward_compatible(self):
        strategy = _make_agg_strategy()
        result = strategy.select_pool(ttft_target=100, itl_target=10, priority=50)
        assert result == 0  # no overrides configured, grid result


# --- AggPoolSelectionStrategy with custom mapping ---


class TestAggPoolSelectionCustomMapping:
    def test_3x3_grid(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=10,
            ttft_max=3010,
            ttft_resolution=3,
            itl_min=10,
            itl_max=100,
            itl_resolution=3,
            agg_pool_mapping=[[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        )
        # Low TTFT, low ITL -> pool 0
        assert strategy.select_pool(ttft_target=100, itl_target=15) == 0
        # Low TTFT, high ITL -> pool 2
        assert strategy.select_pool(ttft_target=100, itl_target=90) == 2
        # Mid TTFT, mid ITL -> pool 2
        assert strategy.select_pool(ttft_target=1500, itl_target=55) == 2
        # High TTFT, low ITL -> pool 2
        assert strategy.select_pool(ttft_target=2500, itl_target=15) == 2


# --- GlobalRouterConfig agg validation tests ---


class TestAggConfigValidation:
    def test_valid_agg_config(self):
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=_make_agg_strategy(),
        )
        config.validate()  # should not raise

    def test_missing_num_agg_pools(self):
        config = GlobalRouterConfig(
            mode="agg",
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=_make_agg_strategy(),
        )
        with pytest.raises(ValueError, match="num_agg_pools required"):
            config.validate()

    def test_missing_namespaces(self):
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_selection_strategy=_make_agg_strategy(),
        )
        with pytest.raises(ValueError, match="agg_pool_dynamo_namespaces required"):
            config.validate()

    def test_missing_strategy(self):
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
        )
        with pytest.raises(ValueError, match="agg_pool_selection_strategy required"):
            config.validate()

    def test_namespace_count_mismatch(self):
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=3,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=_make_agg_strategy(),
        )
        with pytest.raises(ValueError, match="num_agg_pools.*does not match"):
            config.validate()

    def test_invalid_pool_idx_in_mapping(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=10,
            ttft_max=3000,
            ttft_resolution=2,
            itl_min=5,
            itl_max=200,
            itl_resolution=2,
            agg_pool_mapping=[[0, 5], [0, 1]],  # pool 5 is out of range
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="Invalid agg pool index"):
            config.validate()

    def test_mapping_row_count_mismatch(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=10,
            ttft_max=3000,
            ttft_resolution=3,  # expects 3 rows
            itl_min=5,
            itl_max=200,
            itl_resolution=2,
            agg_pool_mapping=[[0, 1], [0, 1]],  # only 2 rows
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="agg_pool_mapping rows.*does not match"):
            config.validate()

    def test_mapping_col_count_mismatch(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=10,
            ttft_max=3000,
            ttft_resolution=2,
            itl_min=5,
            itl_max=200,
            itl_resolution=3,  # expects 3 columns
            agg_pool_mapping=[[0, 1], [0, 1]],  # only 2 columns per row
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="agg_pool_mapping row.*does not match"):
            config.validate()

    def test_priority_override_invalid_target(self):
        strategy = _make_agg_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=1, max_priority=10, target_pool=5)
            ]
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="invalid target_pool"):
            config.validate()

    def test_priority_override_inverted_range(self):
        strategy = _make_agg_strategy(
            priority_overrides=[
                PriorityPoolOverride(min_priority=20, max_priority=5, target_pool=1)
            ]
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="min_priority"):
            config.validate()

    def test_unknown_mode(self):
        config = GlobalRouterConfig(mode="invalid")
        with pytest.raises(ValueError, match="Unknown mode"):
            config.validate()

    def test_ttft_range_invalid(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=3000,
            ttft_max=10,  # min > max
            ttft_resolution=2,
            itl_min=5,
            itl_max=200,
            itl_resolution=2,
            agg_pool_mapping=[[0, 1], [0, 1]],
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="ttft_min.*must be less than"):
            config.validate()

    def test_itl_range_invalid(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min=10,
            ttft_max=3000,
            ttft_resolution=2,
            itl_min=200,
            itl_max=5,  # min > max
            itl_resolution=2,
            agg_pool_mapping=[[0, 1], [0, 1]],
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="itl_min.*must be less than"):
            config.validate()


# --- Config loading tests ---


class TestLoadAggConfig:
    def test_load_agg_config(self, tmp_path):
        config_data = _agg_base_config()
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert config.mode == "agg"
        assert config.num_agg_pools == 2
        assert config.agg_pool_dynamo_namespaces == ["ns-agg-0", "ns-agg-1"]
        assert config.agg_pool_selection_strategy is not None
        assert config.agg_pool_selection_strategy.ttft_min == 10
        assert config.agg_pool_selection_strategy.ttft_max == 3000
        assert config.agg_pool_selection_strategy.itl_min == 5
        assert config.agg_pool_selection_strategy.itl_max == 200

    def test_agg_config_with_priority_overrides(self, tmp_path):
        config_data = _agg_base_config()
        config_data["agg_pool_selection_strategy"]["priority_overrides"] = [
            {"min_priority": 10, "max_priority": 100, "target_pool": 1}
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert len(config.agg_pool_selection_strategy.priority_overrides) == 1
        override = config.agg_pool_selection_strategy.priority_overrides[0]
        assert override.min_priority == 10
        assert override.max_priority == 100
        assert override.target_pool == 1

    def test_agg_config_without_priority_overrides(self, tmp_path):
        config_data = _agg_base_config()
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert config.agg_pool_selection_strategy.priority_overrides == []

    def test_unknown_mode_in_config(self, tmp_path):
        config_data = {"mode": "invalid"}
        config_path = _write_config(tmp_path, config_data)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_config(config_path)

    def test_agg_pool_selection_routes_correctly(self, tmp_path):
        """End-to-end: load config, then verify pool selection works."""
        config_data = _agg_base_config()
        config_data["agg_pool_selection_strategy"]["priority_overrides"] = [
            {"min_priority": 50, "max_priority": 100, "target_pool": 0}
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        strategy = config.agg_pool_selection_strategy
        # Tight TTFT + tight ITL -> pool 0 from grid
        assert strategy.select_pool(ttft_target=100, itl_target=10) == 0
        # Relaxed TTFT + relaxed ITL -> pool 1 from grid
        assert strategy.select_pool(ttft_target=2000, itl_target=150) == 1
        # Priority override: relaxed would be pool 1, but priority 75 -> pool 0
        assert strategy.select_pool(ttft_target=2000, itl_target=150, priority=75) == 0
