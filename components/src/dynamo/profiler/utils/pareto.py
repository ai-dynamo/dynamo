# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from enum import Enum
from itertools import groupby
from typing import Sequence


class ParetoDirection(str, Enum):
    """Optimization direction for one Pareto objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def _direction_multiplier(direction: ParetoDirection | str) -> float:
    try:
        resolved = ParetoDirection(direction)
    except ValueError as error:
        valid = ", ".join(value.value for value in ParetoDirection)
        raise ValueError(
            f"invalid Pareto direction {direction!r}; expected one of: {valid}"
        ) from error
    return 1.0 if resolved is ParetoDirection.MAXIMIZE else -1.0


def _valid_points(
    objectives: Sequence[Sequence[float]],
    directions: Sequence[ParetoDirection | str],
) -> list[tuple[tuple[float, ...], int]]:
    if len(objectives) != len(directions):
        raise ValueError("objectives and directions must have the same length")
    if len(objectives) == 0:
        return []

    row_count = len(objectives[0])
    if any(len(objective) != row_count for objective in objectives[1:]):
        raise ValueError("all objective columns must have the same length")

    multipliers = tuple(_direction_multiplier(direction) for direction in directions)
    points: list[tuple[tuple[float, ...], int]] = []
    for index in range(row_count):
        try:
            values = tuple(
                float(objective[index]) * multiplier
                for objective, multiplier in zip(objectives, multipliers, strict=True)
            )
        except (OverflowError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            points.append((values, index))
    return points


def _compute_pareto_2d(
    points: Sequence[tuple[tuple[float, ...], int]],
    *,
    keep_equivalent: bool,
) -> list[int]:
    ordered = sorted(
        points,
        key=lambda point: (-point[0][0], -point[0][1], point[1]),
    )
    frontier: list[int] = []
    best_second = float("-inf")
    for _, group_iter in groupby(ordered, key=lambda point: point[0][0]):
        group = list(group_iter)
        group_best_second = group[0][0][1]
        if group_best_second <= best_second:
            continue

        equivalents = [
            index for values, index in group if values[1] == group_best_second
        ]
        frontier.extend(equivalents if keep_equivalent else equivalents[:1])
        best_second = group_best_second
    return sorted(frontier)


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _compute_pareto_nd(
    points: Sequence[tuple[tuple[float, ...], int]],
    *,
    keep_equivalent: bool,
) -> list[int]:
    ordered = sorted(points, key=lambda point: (point[0], -point[1]), reverse=True)
    frontier_values: list[tuple[float, ...]] = []
    frontier_indices: list[int] = []
    for values, group_iter in groupby(ordered, key=lambda point: point[0]):
        if any(_dominates(existing, values) for existing in frontier_values):
            continue
        indices = [index for _, index in group_iter]
        frontier_values.append(values)
        frontier_indices.extend(indices if keep_equivalent else indices[:1])
    return sorted(frontier_indices)


def compute_pareto(
    objectives: Sequence[Sequence[float]],
    directions: Sequence[ParetoDirection | str],
    *,
    keep_equivalent: bool = True,
) -> list[int]:
    """Return input indices for the exact non-dominated objective rows.

    Each entry in ``objectives`` is one objective column. Values are normalized
    to finite Python floats, and ``directions`` declares whether the corresponding
    column is minimized or maximized. Rows that cannot be normalized are excluded.

    Equivalent objective rows are all non-dominated mathematically. Keep them by
    default, or set ``keep_equivalent=False`` to retain the first input row only.
    Returned indices are ordered by their position in the input.

    The two-objective path is O(n log n). The generic path compares against the
    current unique frontier and can be quadratic when many N-D rows are
    non-dominated.
    """

    points = _valid_points(objectives, directions)
    if len(objectives) == 2:
        return _compute_pareto_2d(
            points,
            keep_equivalent=keep_equivalent,
        )
    return _compute_pareto_nd(
        points,
        keep_equivalent=keep_equivalent,
    )
