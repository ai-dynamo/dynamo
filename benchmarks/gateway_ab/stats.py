# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics for the Frontend vs Gateway+EPP A/B benchmark. Standard library only.

The unit of replication is a *round*: one aiperf run per side at one
concurrency. Rounds differ by more than requests within a round do (warm caches,
node noise, neighbours), so confidence intervals come from a two-level
bootstrap — resample rounds, then requests within each chosen round. A plain
per-request bootstrap would treat 1,000 requests from three runs as 1,000
independent draws and report intervals far narrower than the run-to-run noise.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Literal, Sequence, Union

# A summary statistic: a percentile in [0, 100], or the mean.
Stat = Union[float, Literal["mean"]]

# Per-request latency metrics, as aiperf names them, and the label to report.
LATENCY_METRICS = {
    "time_to_first_token": "TTFT",
    "inter_token_latency": "ITL",
    "request_latency": "E2E latency",
}
# Per-run scalar metrics from aiperf's summary export.
THROUGHPUT_METRICS = {
    "output_token_throughput": "Output tok/s",
    "request_throughput": "Requests/s",
}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]. Matches numpy's default."""
    if not values:
        raise ValueError("percentile of empty sequence")
    return _percentile_sorted(sorted(values), q)


def _stat(data: Sequence[float], q: Stat) -> float:
    """``q`` of already-sorted ``data``: a percentile, or the mean."""
    return statistics.fmean(data) if q == "mean" else _percentile_sorted(data, q)


def _percentile_sorted(data: Sequence[float], q: float) -> float:
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def pct_delta(a: float, b: float) -> float:
    """Relative change of B against A, in percent."""
    if a == 0:
        return math.inf if b > 0 else (-math.inf if b < 0 else 0.0)
    return (b - a) / a * 100.0


def hierarchical_bootstrap_delta(
    rounds_a: Sequence[Sequence[float]],
    rounds_b: Sequence[Sequence[float]],
    q: float,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for ``pct_delta(pXX(A), pXX(B))``."""
    return hierarchical_bootstrap_deltas(
        rounds_a, rounds_b, (q,), iterations, confidence, seed
    )[q]


def hierarchical_bootstrap_deltas(
    rounds_a: Sequence[Sequence[float]],
    rounds_b: Sequence[Sequence[float]],
    quantiles: Sequence[Stat],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[Stat, tuple[float, float]]:
    """Confidence intervals for B's percent delta at each statistic.

    A statistic is a percentile in [0, 100] or ``"mean"``.

    Each iteration resamples rounds with replacement, then requests with
    replacement within each chosen round, independently for A and B. Every
    quantile is read from the same resample, so one pass serves them all.
    """
    rounds_a = [r for r in rounds_a if r]
    rounds_b = [r for r in rounds_b if r]
    if not rounds_a or not rounds_b:
        raise ValueError("both sides need at least one non-empty round")
    rng = random.Random(seed)
    deltas: dict[Stat, list[float]] = {q: [] for q in quantiles}
    for _ in range(iterations):
        a = sorted(_resample(rng, rounds_a))
        b = sorted(_resample(rng, rounds_b))
        for q in quantiles:
            d = pct_delta(_stat(a, q), _stat(b, q))
            if math.isfinite(d):
                deltas[q].append(d)
    tail = (1.0 - confidence) / 2.0 * 100.0
    return {
        q: (percentile(ds, tail), percentile(ds, 100.0 - tail))
        if ds
        else (math.nan, math.nan)
        for q, ds in deltas.items()
    }


def _resample(rng: random.Random, rounds: Sequence[Sequence[float]]) -> list[float]:
    pooled: list[float] = []
    for chosen in rng.choices(rounds, k=len(rounds)):
        pooled.extend(rng.choices(chosen, k=len(chosen)))
    return pooled


def verdict(
    ci: tuple[float, float], min_effect_pct: float, higher_is_better: bool
) -> str:
    """Classify a B-vs-A delta from its confidence interval.

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
        return "B better" if higher_is_better else "B worse"
    if hi < -min_effect_pct:
        return "B worse" if higher_is_better else "B better"
    if lo > 0 or hi < 0:
        return (
            "B slightly better" if (lo > 0) == higher_is_better else "B slightly worse"
        )
    if lo >= -min_effect_pct and hi <= min_effect_pct:
        return "same"
    return "inconclusive"


def compare_latency(
    rounds_a: Sequence[Sequence[float]],
    rounds_b: Sequence[Sequence[float]],
    quantiles: Sequence[Stat] = ("mean", 50.0, 90.0, 99.0),
    iterations: int = 1000,
    min_effect_pct: float = 2.0,
    seed: int = 0,
) -> dict[str, dict[str, float | str | tuple[float, float]]]:
    """Per-statistic comparison of one latency metric. Lower is better.

    The mean is reported next to the percentiles because latency under
    batching is often multi-modal, and a percentile that sits between two
    modes jumps from one to the other on small shifts in the mix.
    """
    pooled_a = [v for r in rounds_a for v in r]
    pooled_b = [v for r in rounds_b for v in r]
    cis = hierarchical_bootstrap_deltas(
        rounds_a, rounds_b, quantiles, iterations=iterations, seed=seed
    )
    out: dict[str, dict[str, float | str | tuple[float, float]]] = {}
    for q in quantiles:
        a = _stat(sorted(pooled_a), q)
        b = _stat(sorted(pooled_b), q)
        ci = cis[q]
        out[q if q == "mean" else f"p{q:g}"] = {
            "a": a,
            "b": b,
            "delta_pct": pct_delta(a, b),
            "ci_pct": ci,
            "verdict": verdict(ci, min_effect_pct, higher_is_better=False),
        }
    return out


def compare_scalar(
    values_a: Sequence[float],
    values_b: Sequence[float],
    higher_is_better: bool = True,
    min_effect_pct: float = 2.0,
) -> dict[str, float | str | tuple[float, float]]:
    """Compare a per-run scalar (throughput) across rounds.

    Too few rounds for a bootstrap to mean much, so the interval is the
    range of the pairwise deltas, which stays conservative at n=3.
    """
    if not values_a or not values_b:
        raise ValueError("both sides need at least one value")
    a = statistics.median(values_a)
    b = statistics.median(values_b)
    deltas = [pct_delta(x, y) for x in values_a for y in values_b]
    finite = [d for d in deltas if math.isfinite(d)]
    ci = (min(finite), max(finite)) if finite else (math.nan, math.nan)
    return {
        "a": a,
        "b": b,
        "delta_pct": pct_delta(a, b),
        "ci_pct": ci,
        "verdict": verdict(ci, min_effect_pct, higher_is_better),
        "a_cv_pct": _cv(values_a),
        "b_cv_pct": _cv(values_b),
    }


def _cv(values: Sequence[float]) -> float:
    """Coefficient of variation across rounds, in percent: the A/A noise floor."""
    if len(values) < 2:
        return math.nan
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean * 100.0 if mean else math.nan
