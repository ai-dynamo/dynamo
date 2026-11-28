#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for uncached-token-based pool routing in the global router."""

import json
from pathlib import Path

import pytest

from dynamo.global_router.pool_selection import (
    AggPoolSelectionStrategy,
    DecodePoolSelectionStrategy,
    GlobalRouterConfig,
    PrefillPoolSelectionStrategy,
    PriorityPoolOverride,
    UncachedTokensPoolOverride,
    _apply_uncached_overrides,
    load_config,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]

# --- _apply_uncached_overrides tests ---


class TestApplyUncachedOverrides:
    def test_no_estimate_returns_base(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)]
        assert _apply_uncached_overrides(0, None, rules) == 0

    def test_no_rules_returns_base(self):
        assert _apply_uncached_overrides(0, 150000, []) == 0

    def test_match_returns_target(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=2)]
        assert _apply_uncached_overrides(0, 131072, rules) == 2

    def test_below_min_returns_base(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)]
        assert _apply_uncached_overrides(0, 99999, rules) == 0

    def test_bounded_rule_upper_bound_inclusive(self):
        rules = [
            UncachedTokensPoolOverride(
                min_uncached_tokens=100000, max_uncached_tokens=200000, target_pool=1
            )
        ]
        assert _apply_uncached_overrides(0, 200000, rules) == 1
        assert _apply_uncached_overrides(0, 200001, rules) == 0

    def test_unbounded_rule_matches_any_large_value(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)]
        assert _apply_uncached_overrides(0, 10**9, rules) == 1

    def test_first_match_wins(self):
        rules = [
            UncachedTokensPoolOverride(
                min_uncached_tokens=100000, max_uncached_tokens=150000, target_pool=1
            ),
            UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=2),
        ]
        # 120000 matches both rules; first should win
        assert _apply_uncached_overrides(0, 120000, rules) == 1
        # 180000 only matches the second rule
        assert _apply_uncached_overrides(0, 180000, rules) == 2

    def test_zero_estimate_not_matched_by_positive_min(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)]
        # Explicit 0 is a valid estimate (fully cached expectation), not "no
        # estimate"; with a positive threshold it simply does not match.
        assert _apply_uncached_overrides(0, 0, rules) == 0

    def test_zero_min_matches_zero_estimate(self):
        rules = [UncachedTokensPoolOverride(min_uncached_tokens=0, target_pool=1)]
        assert _apply_uncached_overrides(0, 0, rules) == 1


# --- PrefillPoolSelectionStrategy with uncached overrides ---


def _make_prefill_strategy(
    uncached_overrides=None,
) -> PrefillPoolSelectionStrategy:
    return PrefillPoolSelectionStrategy(
        ttft_min_ms=10,
        ttft_max_ms=3000,
        ttft_resolution=2,
        isl_min=0,
        isl_max=32000,
        isl_resolution=2,
        prefill_pool_mapping=[[0, 1], [0, 1]],
        uncached_overrides=uncached_overrides or [],
    )


class TestPrefillSelectPoolWithUncached:
    def test_uncached_override_takes_precedence_over_grid(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)
            ]
        )
        # Grid would select pool 0 (low ISL, low TTFT), but the uncached
        # override redirects to the compute pool.
        result = strategy.select_pool(
            isl=100, ttft_target_ms=50, uncached_tokens=131072
        )
        assert result == 1

    def test_no_estimate_uses_grid(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)
            ]
        )
        result = strategy.select_pool(isl=100, ttft_target_ms=50)
        assert result == 0

    def test_small_estimate_uses_grid(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)
            ]
        )
        # 120K-token agent multi-turn request that declares only 2K uncached:
        # high ISL grid bucket, but the small uncached estimate keeps it on
        # the cache-friendly pool.
        result = strategy.select_pool(isl=100, ttft_target_ms=50, uncached_tokens=2048)
        assert result == 0

    def test_priority_override_wins_over_uncached_override(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1)
            ]
        )
        strategy.priority_overrides = [
            PriorityPoolOverride(min_priority=10, max_priority=100, target_pool=0)
        ]
        result = strategy.select_pool(
            isl=100, ttft_target_ms=50, priority=50, uncached_tokens=131072
        )
        assert result == 0

    def test_no_overrides_backward_compatible(self):
        strategy = _make_prefill_strategy()
        result = strategy.select_pool(
            isl=100, ttft_target_ms=50, uncached_tokens=131072
        )
        assert result == 0


# --- AggPoolSelectionStrategy with uncached overrides ---


def _make_agg_strategy(uncached_overrides=None) -> AggPoolSelectionStrategy:
    return AggPoolSelectionStrategy(
        ttft_min_ms=10,
        ttft_max_ms=3000,
        ttft_resolution=1,
        itl_min_ms=10,
        itl_max_ms=500,
        itl_resolution=1,
        agg_pool_mapping=[[0]],
        uncached_overrides=uncached_overrides or [],
    )


class TestAggSelectPoolWithUncached:
    def test_uncached_override_takes_precedence(self):
        strategy = _make_agg_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=0)
            ]
        )
        # Single-cell grid maps to pool 0 anyway; use two pools to observe it.
        strategy.agg_pool_mapping = [[1]]
        result = strategy.select_pool(
            ttft_target_ms=100, itl_target_ms=50, uncached_tokens=131072
        )
        assert result == 0

    def test_no_estimate_uses_grid(self):
        strategy = _make_agg_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=0)
            ]
        )
        strategy.agg_pool_mapping = [[1]]
        result = strategy.select_pool(ttft_target_ms=100, itl_target_ms=50)
        assert result == 1


# --- Config loading tests ---


def _write_config(tmp_dir: Path, config_data: dict) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _base_disagg_config() -> dict:
    return {
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


def _base_agg_config() -> dict:
    return {
        "mode": "agg",
        "num_agg_pools": 2,
        "agg_pool_dynamo_namespaces": ["ns-agg-0", "ns-agg-1"],
        "agg_pool_selection_strategy": {
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 2,
            "itl_min": 10,
            "itl_max": 500,
            "itl_resolution": 2,
            "agg_pool_mapping": [[0, 1], [0, 1]],
        },
    }


class TestLoadConfigWithUncachedOverrides:
    def test_loads_disagg_prefill_uncached_overrides(self, tmp_path):
        config_data = _base_disagg_config()
        config_data["prefill_pool_selection_strategy"]["uncached_overrides"] = [
            {"min_uncached_tokens": 100000, "target_pool": 1}
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert len(config.prefill_pool_selection_strategy.uncached_overrides) == 1
        override = config.prefill_pool_selection_strategy.uncached_overrides[0]
        assert override.min_uncached_tokens == 100000
        assert override.max_uncached_tokens is None
        assert override.target_pool == 1

    def test_loads_disagg_prefill_bounded_uncached_override(self, tmp_path):
        config_data = _base_disagg_config()
        config_data["prefill_pool_selection_strategy"]["uncached_overrides"] = [
            {
                "min_uncached_tokens": 100000,
                "max_uncached_tokens": 200000,
                "target_pool": 1,
            }
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        override = config.prefill_pool_selection_strategy.uncached_overrides[0]
        assert override.max_uncached_tokens == 200000

    def test_loads_agg_uncached_overrides(self, tmp_path):
        config_data = _base_agg_config()
        config_data["agg_pool_selection_strategy"]["uncached_overrides"] = [
            {"min_uncached_tokens": 100000, "target_pool": 1}
        ]
        config_path = _write_config(tmp_path, config_data)
        config = load_config(config_path)

        assert len(config.agg_pool_selection_strategy.uncached_overrides) == 1
        assert config.agg_pool_selection_strategy.uncached_overrides[0].target_pool == 1

    def test_without_uncached_overrides_defaults_empty(self, tmp_path):
        config_path = _write_config(tmp_path, _base_disagg_config())
        config = load_config(config_path)

        assert config.prefill_pool_selection_strategy.uncached_overrides == []
        assert config.decode_pool_selection_strategy.priority_overrides == []


# --- Validation tests ---


class TestValidateUncachedOverrides:
    def _make_config(self, strategy) -> GlobalRouterConfig:
        return GlobalRouterConfig(
            mode="disagg",
            num_prefill_pools=2,
            num_decode_pools=2,
            prefill_pool_dynamo_namespaces=["a", "b"],
            decode_pool_dynamo_namespaces=["c", "d"],
            prefill_pool_selection_strategy=strategy,
            decode_pool_selection_strategy=DecodePoolSelectionStrategy(
                itl_min_ms=10,
                itl_max_ms=500,
                itl_resolution=2,
                context_length_min=0,
                context_length_max=32000,
                context_length_resolution=2,
                decode_pool_mapping=[[0, 1], [0, 1]],
            ),
        )

    def test_invalid_target_pool(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=0, target_pool=5)
            ]
        )
        config = self._make_config(strategy)
        with pytest.raises(ValueError, match="invalid target_pool"):
            config.validate()

    def test_negative_min_uncached_tokens(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=-1, target_pool=1)
            ]
        )
        config = self._make_config(strategy)
        with pytest.raises(ValueError, match="must be non-negative"):
            config.validate()

    def test_max_less_than_min(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(
                    min_uncached_tokens=100000,
                    max_uncached_tokens=50000,
                    target_pool=1,
                )
            ]
        )
        config = self._make_config(strategy)
        with pytest.raises(ValueError, match="must be >="):
            config.validate()

    def test_valid_overrides_pass(self):
        strategy = _make_prefill_strategy(
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=100000, target_pool=1),
                UncachedTokensPoolOverride(
                    min_uncached_tokens=200000,
                    max_uncached_tokens=400000,
                    target_pool=0,
                ),
            ]
        )
        config = self._make_config(strategy)
        config.validate()  # should not raise

    def test_agg_invalid_target_pool(self):
        strategy = AggPoolSelectionStrategy(
            ttft_min_ms=10,
            ttft_max_ms=3000,
            ttft_resolution=1,
            itl_min_ms=10,
            itl_max_ms=500,
            itl_resolution=1,
            agg_pool_mapping=[[0]],
            uncached_overrides=[
                UncachedTokensPoolOverride(min_uncached_tokens=0, target_pool=3)
            ],
        )
        config = GlobalRouterConfig(
            mode="agg",
            num_agg_pools=2,
            agg_pool_dynamo_namespaces=["a", "b"],
            agg_pool_selection_strategy=strategy,
        )
        with pytest.raises(ValueError, match="invalid target_pool"):
            config.validate()
