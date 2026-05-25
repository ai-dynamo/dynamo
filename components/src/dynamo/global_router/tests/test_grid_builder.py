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
    # Latency bands use the strict LOWER edge: col0=[0,200)->target 0ms, col1=[200,400)->target 200ms.
    # short band (ISL 1000: tp1=250,tp2=140,tp4=70): tight(0ms) nobody meets -> fastest tp4;
    #                                                loose(200ms) -> cheapest meeting = tp2.
    # long band  (ISL 3000: tp1=650,tp2=340,tp4=150): tight(0ms) -> fastest tp4; loose(200ms) -> only tp4.
    assert grid == [[2, 1], [2, 2]]


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
    assert strat.select_pool(isl=1000, ttft_target_ms=300) == 1  # short, loose col -> tp2
    assert strat.select_pool(isl=1000, ttft_target_ms=150) == 2  # short, tight col -> tp4 (fastest)
    assert strat.select_pool(isl=3000, ttft_target_ms=150) == 2  # long,  tight col -> tp4
    assert strat.select_pool(isl=3000, ttft_target_ms=400) == 2  # long,  loose col -> tp4


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
