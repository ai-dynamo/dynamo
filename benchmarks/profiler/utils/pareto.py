# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def compute_pareto(x, y):
    """
    Compute the pareto front (top-left is better) for the given x and y values.

    Returns:
        tuple: (xs, ys, indices) where:
            - xs: list of x values on the pareto front
            - ys: list of y values on the pareto front
            - indices: list of original indices corresponding to the pareto points
    """
    # Validate inputs
    if x is None or y is None:
        return [], [], []

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) == 0:
        return [], [], []

    # Build point list with original indices and sort by x asc, then y desc
    points = [(x[i], y[i], i) for i in range(len(x))]
    points.sort(key=lambda p: (p[0], -p[1]))

    # Single pass to keep only non-dominated points (minimize x, maximize y)
    pareto = []
    max_y = float("-inf")
    for px, py, idx in points:
        if py > max_y:
            pareto.append((px, py, idx))
            max_y = py

    # Return sorted by x ascending for convenience
    pareto.sort(key=lambda p: (p[0], p[1]))
    xs = [px for px, _, _ in pareto]
    ys = [py for _, py, _ in pareto]
    indices = [idx for _, _, idx in pareto]
    return xs, ys, indices
