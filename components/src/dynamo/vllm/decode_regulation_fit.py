# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit the decode-regulation step-time model from a scheduler trace.

Reads DYN_DECODE_REGULATION_TRACE files (one JSON record per step) written in
any mode other than off; DYN_DECODE_REGULATION_MODE=observe records a run
without skipping anything. Fits T(n, L) = a + b*n + c*L ms by least squares on
decode-only steps (n decodes, L their total context in tokens), reports the
error by batch size, and prints the DYN_DECODE_REGULATION_COEFFS line to use.

    python3 -m dynamo.vllm.decode_regulation_fit trace.jsonl [trace2.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import sys

Row = tuple[float, float, float]  # (n, L, step ms)


def load(path: str, skip_first_s: float, max_ms: float) -> list[Row]:
    rows: list[Row] = []
    t0 = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a torn last line of a live trace
                continue
            t0 = r["t"] if t0 is None else t0
            if r["t"] - t0 < skip_first_s or not r["decode_only"] or r["n"] < 1:
                continue
            if 0 < r["actual"] <= max_ms:  # slower steps are hiccups, not load
                rows.append((float(r["n"]), float(r["ctx"]), float(r["actual"])))
    return rows


def fit(rows: list[Row]) -> tuple[float, float, float]:
    """Least squares on [1, n, L / 1000] (scaled for conditioning)."""
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for n, ctx, y in rows:
        x = (1.0, n, ctx / 1000.0)
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    a = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        if abs(a[col][col]) < 1e-12:
            raise ValueError("need steps with varied batch size and context")
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for k in range(col, 4):
                a[r][k] -= f * a[col][k]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (a[r][3] - sum(a[r][k] * x[k] for k in range(r + 1, 3))) / a[r][r]
    return x[0], x[1], x[2] / 1000.0


def mape(c: tuple[float, float, float], rows: list[Row]) -> float:
    def pred(n: float, ctx: float) -> float:
        return max(0.0, c[0] + c[1] * n + c[2] * ctx)

    return sum(abs(pred(n, ctx) - y) / y for n, ctx, y in rows) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("traces", nargs="+", help="DYN_DECODE_REGULATION_TRACE files")
    ap.add_argument(
        "--skip-first-s", type=float, default=60.0, help="warm-up to drop per file"
    )
    ap.add_argument(
        "--max-ms", type=float, default=200.0, help="drop steps slower than this"
    )
    a = ap.parse_args()
    rows = [r for p in a.traces for r in load(p, a.skip_first_s, a.max_ms)]
    if len(rows) < 3:
        sys.exit("not enough decode-only steps; record a longer trace under load")
    c = fit(rows)
    print(f"{len(rows)} decode-only steps")
    print(f"T(n, L) = {c[0]:.3f} + {c[1]:.4f}*n + {c[2] * 1000:.4f}*(L/1k) ms")
    print(f"mean absolute percentage error: {mape(c, rows):.1%}")
    for lo, hi in ((1, 8), (8, 32), (32, 128), (128, 10**9)):
        rs = [r for r in rows if lo <= r[0] < hi]
        if rs:
            print(
                f"  n {lo}-{min(hi - 1, 9999)}: steps={len(rs)} error={mape(c, rs):.1%}"
            )
    print(f"DYN_DECODE_REGULATION_COEFFS={c[0]:.4f},{c[1]:.6f},{c[2]:.9f}")


if __name__ == "__main__":
    main()
