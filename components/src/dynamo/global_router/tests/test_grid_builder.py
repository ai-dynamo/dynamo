# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for build_isl_latency_grid (Phase 3): turning per-pool latency
curves into the (size x latency) -> pool-index grid that both the Global Router
and the EPP pool-selector consume. Includes a build->select parity check."""

import pytest

from dynamo.global_router.pool_selection import (
    PrefillPoolSelectionStrategy,
    build_isl_latency_grid,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


# 3 pools: tp1(cost1), tp2(cost2), tp4(cost4). Latency = base + slope*ISL;
# bigger pools are faster, especially at long ISL.
_CURVES = [
    lambda isl: 50 + 0.20 * isl,  # tp1: cheap, slow on long prompts
    lambda isl: 40 + 0.10 * isl,  # tp2
    lambda isl: 30 + 0.04 * isl,  # tp4: fast even on long prompts
]
_COSTS = [1, 2, 4]


def _grid():
    return build_isl_latency_grid(
        _CURVES,
        _COSTS,
        size_min=0,
        size_max=4000,
        size_resolution=2,
        latency_min_ms=0,
        latency_max_ms=400,
        latency_resolution=2,
    )


def test_grid_shape_and_values():
    grid = _grid()
    assert len(grid) == 2 and all(len(r) == 2 for r in grid)
    # short band: tight target -> tp2 (tp1 too slow), loose -> cheapest tp1.
    # long band:  tight -> tp4 (only it meets), loose -> tp2.
    assert grid == [[1, 0], [2, 1]]


def test_build_select_parity():
    """The grid feeds the runtime selection strategy unchanged."""
    grid = _grid()
    strat = PrefillPoolSelectionStrategy(
        ttft_min_ms=0,
        ttft_max_ms=400,
        ttft_resolution=2,
        isl_min=0,
        isl_max=4000,
        isl_resolution=2,
        prefill_pool_mapping=grid,
    )
    assert strat.select_pool(isl=1000, ttft_target_ms=300) == 0  # short, loose -> tp1
    assert strat.select_pool(isl=1000, ttft_target_ms=150) == 1  # short, tight -> tp2
    assert strat.select_pool(isl=3000, ttft_target_ms=150) == 2  # long, tight  -> tp4
    assert strat.select_pool(isl=3000, ttft_target_ms=400) == 1  # long, loose  -> tp2


def test_no_pool_meets_target_picks_fastest():
    # impossibly tight target: pick the fastest (lowest-latency) pool = tp4.
    grid = build_isl_latency_grid(
        _CURVES,
        _COSTS,
        size_min=0,
        size_max=4000,
        size_resolution=1,
        latency_min_ms=0,
        latency_max_ms=1,  # 1ms target: nobody meets it
        latency_resolution=1,
    )
    assert grid == [[2]]


def test_validation_errors():
    with pytest.raises(ValueError):
        build_isl_latency_grid(
            [],
            [],
            size_min=0,
            size_max=1,
            size_resolution=1,
            latency_min_ms=0,
            latency_max_ms=1,
            latency_resolution=1,
        )
    with pytest.raises(ValueError):
        build_isl_latency_grid(
            _CURVES,
            [1, 2],
            size_min=0,
            size_max=1,
            size_resolution=1,
            latency_min_ms=0,
            latency_max_ms=1,
            latency_resolution=1,
        )
