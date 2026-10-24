# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics for the Frontend vs Gateway+EPP A/B benchmark. Standard library only.

The unit of replication is a *round*: one aiperf run per side at one load
point. Within a round both sides replay the same seed back to back, so a round
is a matched pair. Each round reduces to one value per side (aiperf's mean or
percentile for the run, or a throughput), and B is compared with A through a
paired t-interval on the per-round log ratio B/A.

Rounds differ by more than requests within a round do: the seed's arrival
pattern, cache warmth, and node noise move a whole round. Pairing cancels the
part both sides share, and the t-interval keeps its 95% coverage with the three
rounds a run usually has. A per-request interval would treat 1,000 requests from three
runs as 1,000 independent draws and come out far too narrow; a bootstrap over
three rounds is too narrow as well.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

# Per-request latency metrics, as aiperf names them, and the label to report.
LATENCY_METRICS = {
    "time_to_first_token": "TTFT",
    "inter_token_latency": "ITL",
    "request_latency": "E2E latency",
}
# The statistics of each latency metric that are compared, keyed by report
# label, with the field of aiperf's summary export that holds each per run.
LATENCY_STATS = {"mean": "avg", "p50": "p50", "p90": "p90", "p99": "p99"}
# Per-run scalar metrics from aiperf's summary export.
THROUGHPUT_METRICS = {
    "output_token_throughput": "Output tok/s",
    "request_throughput": "Requests/s",
}


def t_quantile(p: float, df: int) -> float:
    """Inverse CDF of Student's t with ``df`` degrees of freedom, ``p`` in (0.5, 1)."""
    if not 0.5 < p < 1.0 or df < 1:
        raise ValueError(f"t_quantile needs 0.5 < p < 1 and df >= 1, got {p}, {df}")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _t_cdf(t: float, df: float) -> float:
    """CDF of Student's t at ``t >= 0``."""
    return 1.0 - 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function, by modified Lentz."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((a - 1.0 + m2) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def paired_change(
    a: Sequence[float], b: Sequence[float], confidence: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """B's percent change against A, from per-round values matched by index.

    Works on the log ratio, so the change is multiplicative and an interval
    never crosses -100%. Returns the point estimate and the paired
    t-interval; with fewer than two pairs the interval is NaN.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need matched, non-empty per-round values")
    logs = [math.log(y / x) for x, y in zip(a, b)]
    mean = statistics.fmean(logs)
    if len(logs) < 2:
        return _pct(mean), (math.nan, math.nan)
    half = (
        t_quantile(0.5 + confidence / 2.0, len(logs) - 1)
        * statistics.stdev(logs)
        / math.sqrt(len(logs))
    )
    return _pct(mean), (_pct(mean - half), _pct(mean + half))


def _pct(log_ratio: float) -> float:
    return math.expm1(log_ratio) * 100.0


def verdict(
    ci: tuple[float, float],
    min_effect_pct: float,
    higher_is_better: bool,
    subject: str = "B",
) -> str:
    """Classify a B-vs-A delta from its confidence interval.

    ``subject`` names B in the verdict.

    - ``B better`` / ``B worse``: the whole interval is beyond
      ``min_effect_pct`` in one direction.
    - ``B slightly better`` / ``B slightly worse``: the interval excludes zero
      but reaches inside ``min_effect_pct``, so the difference is real but may
      be negligible.
    - ``same``: the interval includes zero and lies within
      ``±min_effect_pct``, so any difference is negligible.
    - ``inconclusive``: the interval includes zero but reaches past
      ``min_effect_pct``; the data cannot tell a negligible difference from
      a large one. More rounds or a quieter cluster narrow it.
    """
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "n/a"
    if lo > min_effect_pct:
        return f"{subject} better" if higher_is_better else f"{subject} worse"
    if hi < -min_effect_pct:
        return f"{subject} worse" if higher_is_better else f"{subject} better"
    if lo > 0 or hi < 0:
        better = (lo > 0) == higher_is_better
        return f"{subject} slightly {'better' if better else 'worse'}"
    if lo >= -min_effect_pct and hi <= min_effect_pct:
        return "same"
    return "inconclusive"


def compare_scalar(
    values_a: Mapping[int, float | None],
    values_b: Mapping[int, float | None],
    higher_is_better: bool = True,
    min_effect_pct: float = 2.0,
    subject: str = "B",
) -> dict[str, Any] | None:
    """One comparison row from per-round values, with rounds matched by index.

    A round missing on either side, or with a non-positive or non-finite
    value, has no log ratio and is left out. ``a`` and ``b`` are geometric
    means over the kept rounds, so ``delta_pct`` is exactly B's change
    against A as the columns show it. ``subject`` names B in the verdict.
    """
    kept = [
        (x, y)
        for x, y in (
            (values_a[r], values_b.get(r)) for r in sorted(values_a) if r in values_b
        )
        if x is not None
        and y is not None
        and x > 0
        and y > 0
        and math.isfinite(x)
        and math.isfinite(y)
    ]
    if not kept:
        return None
    a = [x for x, _ in kept]
    b = [y for _, y in kept]
    delta, ci = paired_change(a, b)
    return {
        "a": statistics.geometric_mean(a),
        "b": statistics.geometric_mean(b),
        "delta_pct": delta,
        "ci_pct": ci,
        "verdict": verdict(ci, min_effect_pct, higher_is_better, subject),
        "rounds": len(kept),
    }
