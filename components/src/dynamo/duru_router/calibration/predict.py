# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared calibration and live-cost interpolation, in milliseconds."""

import bisect
import math


def bracket(axis, x, *, extrapolate=False):
    if not axis or (not extrapolate and not axis[0] <= x <= axis[-1]):
        raise ValueError(f"{x} outside calibrated range {axis}")
    if len(axis) == 1:
        return 0, 0, 0.0
    high = min(max(1, bisect.bisect_right(axis, x)), len(axis) - 1)
    low = high - 1
    return low, high, (x - axis[low]) / (axis[high] - axis[low])


def predict_prefill_table(table, p, d, *, extrapolate=False):
    """Interpolate only within an explicitly declared execution regime.

    P_discontinuities contains inclusive upper bounds (e.g. 256 means the
    next regime starts at 257). They must come from runtime evidence, not be
    inferred silently from heldout errors. Near an unmeasured regime edge,
    extend the closest in-regime row constantly; never blend across the jump.
    Outer-range extrapolation is explicit; graph regimes are always respected.
    """
    ps = table["P"]
    i, j, u = bracket(ps, p, extrapolate=extrapolate)
    boundaries = table.get("P_discontinuities", [])
    if boundaries:
        if boundaries != sorted(set(boundaries)) or any(
            type(b) is not int or not ps[0] <= b < ps[-1] for b in boundaries
        ):
            raise ValueError("Invalid prefill execution boundaries")
        region = bisect.bisect_left(boundaries, p)
        first = 0 if region == 0 else bisect.bisect_right(ps, boundaries[region - 1])
        last = (
            len(ps)
            if region == len(boundaries)
            else bisect.bisect_right(ps, boundaries[region])
        )
        if first == last:
            raise ValueError("No training points in prefill execution regime")
        x = (
            p
            if extrapolate and (p < ps[0] or p > ps[-1])
            else min(max(p, ps[first]), ps[last - 1])
        )
        a, b, u = bracket(ps[first:last], x, extrapolate=extrapolate)
        i, j = first + a, first + b
    d_lower, d_upper, v = bracket(table["D"], d, extrapolate=extrapolate)
    z = table["wall_ms"]
    return (1 - u) * ((1 - v) * z[i][d_lower] + v * z[i][d_upper]) + u * (
        (1 - v) * z[j][d_lower] + v * z[j][d_upper]
    )


def predict(table, step):
    """Continue edge slopes across outer ranges, flooring extrapolated costs at zero.

    In-grid interpolation and separation of graph regimes remain unchanged.
    """
    if step["phase"] == "prefill":
        value = predict_prefill_table(table, step["p"], step["d"], extrapolate=True)
        outside = not (
            table["P"][0] <= step["p"] <= table["P"][-1]
            and table["D"][0] <= step["d"] <= table["D"][-1]
        )
    elif step["phase"] == "decode":
        grid = table["grid"]
        axis = [row["K"] for row in grid]
        lo, hi, weight = bracket(axis, step["k"], extrapolate=True)
        value = (1 - weight) * grid[lo]["wall_ms"] + weight * grid[hi]["wall_ms"]
        outside = not axis[0] <= step["k"] <= axis[-1]
    else:
        raise ValueError("Unknown calibration phase")
    if outside:
        value = max(0.0, value)
    return value, outside


def check_tables(tables):
    """Reject malformed measurements; this is not an accuracy validation gate."""
    prefill = tables["prefill"]
    decode = tables["decode"]["grid"]
    for axis in (prefill["P"], prefill["D"], [row["K"] for row in decode]):
        if (
            len(axis) < 2
            or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) for x in axis
            )
            or any(a >= b for a, b in zip(axis, axis[1:]))
        ):
            raise ValueError("Calibration axes must be finite and strictly increasing")
    rows = prefill["wall_ms"]
    if len(rows) != len(prefill["P"]) or any(
        len(row) != len(prefill["D"]) for row in rows
    ):
        raise ValueError("Prefill table dimensions do not match its axes")
    values = [v for row in rows for v in row] + [row["wall_ms"] for row in decode]
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise ValueError("Calibration measurements must be finite and positive")
    for p in [prefill["P"][0], *[b + 1 for b in prefill.get("P_discontinuities", [])]]:
        predict_prefill_table(prefill, p, prefill["D"][0])
