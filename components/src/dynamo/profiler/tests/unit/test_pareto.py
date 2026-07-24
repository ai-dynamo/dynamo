# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import random
from itertools import product
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dynamo.profiler.utils.defaults import DEFAULT_GPU_COST_PER_HOUR
from dynamo.profiler.utils.pareto import ParetoDirection, compute_pareto
from dynamo.profiler.utils.plot import plot_pd_joint_results

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def _brute_force_pareto_indices(
    objectives: list[list[float]],
    directions: tuple[ParetoDirection, ...],
    *,
    keep_equivalent: bool,
) -> list[int]:
    multipliers = [
        1.0 if direction is ParetoDirection.MAXIMIZE else -1.0
        for direction in directions
    ]
    points = [
        tuple(
            objective[index] * multiplier
            for objective, multiplier in zip(objectives, multipliers, strict=True)
        )
        for index in range(len(objectives[0]))
    ]
    frontier = []
    for index, point in enumerate(points):
        dominated = any(
            other_index != index
            and all(a >= b for a, b in zip(other, point, strict=True))
            and any(a > b for a, b in zip(other, point, strict=True))
            for other_index, other in enumerate(points)
        )
        if dominated:
            continue
        if not keep_equivalent and any(points[kept] == point for kept in frontier):
            continue
        frontier.append(index)
    return frontier


def test_compute_pareto_supports_objective_directions():
    objectives = [
        [5.0, 4.0, 3.0, 2.0],
        [5.0, 6.0, 4.0, 7.0],
    ]

    assert compute_pareto(
        objectives,
        [ParetoDirection.MINIMIZE, ParetoDirection.MAXIMIZE],
    ) == [3]
    assert compute_pareto(
        objectives,
        [ParetoDirection.MAXIMIZE, ParetoDirection.MAXIMIZE],
    ) == [0, 1, 3]


def test_compute_pareto_controls_equivalent_rows():
    objectives = [[1.0, 1.0, 2.0], [3.0, 3.0, 2.0]]
    directions = [ParetoDirection.MINIMIZE, ParetoDirection.MAXIMIZE]

    assert compute_pareto(objectives, directions) == [0, 1]
    assert compute_pareto(
        objectives,
        directions,
        keep_equivalent=False,
    ) == [0]


def test_compute_pareto_skips_any_invalid_objective():
    assert compute_pareto(
        [
            [1.0, 2.0, math.nan, 4.0, "not-a-number", 10**400],
            [5.0, math.inf, 3.0, 2.0, 1.0, 0.0],
        ],
        [ParetoDirection.MINIMIZE, ParetoDirection.MAXIMIZE],
    ) == [0]


def test_compute_pareto_supports_empty_and_single_objective_inputs():
    assert compute_pareto([], []) == []
    assert compute_pareto([[]], ["maximize"]) == []
    assert compute_pareto([[3.0, 1.0, 1.0, 2.0]], ["minimize"]) == [1, 2]
    assert compute_pareto(
        [[3.0, 1.0, 1.0, 2.0]],
        ["minimize"],
        keep_equivalent=False,
    ) == [1]
    assert (
        compute_pareto(
            [[math.nan, math.inf]],
            ["maximize"],
        )
        == []
    )


@pytest.mark.parametrize("objective_count", [2, 3, 4])
@pytest.mark.parametrize("keep_equivalent", [False, True])
def test_compute_pareto_matches_brute_force_oracle(
    objective_count: int,
    keep_equivalent: bool,
):
    random_source = random.Random(42 + objective_count)
    objectives = [
        [float(random_source.randrange(6)) for _ in range(80)]
        for _ in range(objective_count)
    ]

    for directions in product(ParetoDirection, repeat=objective_count):
        assert compute_pareto(
            objectives,
            directions,
            keep_equivalent=keep_equivalent,
        ) == _brute_force_pareto_indices(
            objectives,
            directions,
            keep_equivalent=keep_equivalent,
        )


def test_compute_pareto_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="same length"):
        compute_pareto([[1.0]], [])
    with pytest.raises(ValueError, match="same length"):
        compute_pareto([[1.0], [1.0, 2.0]], ["maximize", "maximize"])
    with pytest.raises(ValueError, match="invalid Pareto direction"):
        compute_pareto([[1.0]], ["sideways"])


def test_plot_pd_joint_results_uses_sorted_pareto_pairs(monkeypatch):
    prefill_data = SimpleNamespace(
        ttft=["20", 10.0, 30.0],
        thpt_per_gpu=[200.0, 100.0, 150.0],
    )
    decode_data = SimpleNamespace(
        itl=[3.0, "1", 2.0],
        thpt_per_gpu=[300.0, 100.0, 250.0],
    )
    plotted_lines = []

    def capture_plot(*_args, **_kwargs):
        plotted_lines.extend(
            (
                np.asarray(line.get_xdata(), dtype=float),
                np.asarray(line.get_ydata(), dtype=float),
            )
            for line in plt.gca().lines
        )

    monkeypatch.setattr("matplotlib.pyplot.savefig", capture_plot)

    plot_pd_joint_results(100, 10, prefill_data, decode_data, "/unused")

    decode_rates = np.array([1000.0, 500.0, 1000.0 / 3.0])
    decode_cost = (
        10 * 1000 / np.array([100.0, 250.0, 300.0]) * DEFAULT_GPU_COST_PER_HOUR / 3600
    )
    expected_prefill_costs = [
        100 * 1000 / throughput * DEFAULT_GPU_COST_PER_HOUR / 3600
        for throughput in (100.0, 200.0)
    ]

    assert len(plotted_lines) == 2
    for (actual_x, actual_y), prefill_cost in zip(
        plotted_lines,
        expected_prefill_costs,
        strict=True,
    ):
        np.testing.assert_allclose(actual_x, decode_rates)
        np.testing.assert_allclose(actual_y, decode_cost + prefill_cost)
