# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.profiler.utils.pareto import compute_pareto

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def test_compute_pareto_skips_nan_points():
    xs, ys, indices = compute_pareto(
        [float("nan"), 10.0, 20.0],
        [100.0, float("nan"), 200.0],
    )

    assert xs == [20.0]
    assert ys == [200.0]
    assert indices == [2]


def test_compute_pareto_returns_empty_when_all_points_are_invalid():
    assert compute_pareto([float("nan")], [float("nan")]) == ([], [], [])
