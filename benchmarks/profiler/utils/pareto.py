# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def compute_pareto(x, y):
    """
    compute the pareto front (top-left is better) for the given x and y values
    return sorted lists of the x and y values for the pareto front
    """
    # Validate inputs
    if x is None or y is None:
        return [], []

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) == 0:
        return [], []

    # Build point list and sort by x asc, then y desc so we prefer smaller x and larger y.
    points = list(zip(x, y))
    points.sort(key=lambda p: (p[0], -p[1]))

    # Single pass to keep only non-dominated points (minimize x, maximize y).
    pareto = []
    max_y = float("-inf")
    for px, py in points:
        if py > max_y:
            pareto.append((px, py))
            max_y = py

    # Return sorted by x ascending for convenience
    pareto.sort(key=lambda p: (p[0], p[1]))
    xs = [px for px, _ in pareto]
    ys = [py for _, py in pareto]
    return xs, ys
