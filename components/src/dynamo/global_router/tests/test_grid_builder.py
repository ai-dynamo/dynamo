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
    lambda isl: 50 + 0.05 * isl,  # tp1: cheap, slower on long prompts
    lambda isl: 40 + 0.03 * isl,  # tp2
    lambda isl: 30 + 0.02 * isl,  # tp4: fast even on long prompts
]
_COSTS = [1, 2, 4]


def _grid():
    return build_isl_latency_grid(
        _CURVES,
        _COSTS,
        size_min=0,
        size_max=4000,
        size_resolution=2,
        latency_min_ms=100,
        latency_max_ms=500,
        latency_resolution=2,
    )


def test_grid_shape_and_values():
    grid = _grid()
    assert len(grid) == 2 and all(len(r) == 2 for r in grid)
    # Worst-case corners: size axis = UPPER edge (col->2000, 4000 ISL), latency axis =
    # LOWER edge (col0->100ms tight, col1->300ms loose).
    # short band (ISL 2000: tp1=150,tp2=100,tp4=70): tight(100ms) -> cheapest meeting tp2;
    #                                                loose(300ms) -> cheapest tp1.
    # long band  (ISL 4000: tp1=250,tp2=160,tp4=110): tight(100ms) nobody meets -> fastest tp4;
    #                                                loose(300ms) -> cheapest tp1.
    assert grid == [[1, 0], [2, 0]]


def test_build_select_parity():
    """The grid feeds the runtime selection strategy unchanged."""
    grid = _grid()
    strat = PrefillPoolSelectionStrategy(
        ttft_min_ms=100,
        ttft_max_ms=500,
        ttft_resolution=2,
        isl_min=0,
        isl_max=4000,
        isl_resolution=2,
        prefill_pool_mapping=grid,
    )
    assert strat.select_pool(isl=1000, ttft_target_ms=300) == 0  # short, loose col -> tp1
    assert strat.select_pool(isl=1000, ttft_target_ms=150) == 1  # short, tight col -> tp2
    assert strat.select_pool(isl=3000, ttft_target_ms=150) == 2  # long,  tight col -> tp4
    assert strat.select_pool(isl=3000, ttft_target_ms=400) == 0  # long,  loose col -> tp1


def test_no_pool_meets_target_picks_fastest():
    # impossibly tight target (lower edge 0ms): nobody meets -> fastest pool = tp4.
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
