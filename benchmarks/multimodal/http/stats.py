# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate ``RunResult`` samples into a ``Summary`` (avg, p50, p90, p99).

Latency stats are computed over success samples only — including timeout
or status-error latencies would skew the comparison toward client-side
wait time rather than server work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .runner import RunResult


@dataclass
class Summary:
    backend: str
    n: int
    wall_s: float
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    outcomes: Counter[str]


def _percentile_ms(sorted_seconds: list[float], q: float) -> float:
    if not sorted_seconds:
        return 0.0
    idx = min(int(q * (len(sorted_seconds) - 1)), len(sorted_seconds) - 1)
    return sorted_seconds[idx] * 1000.0


def summarize(result: RunResult) -> Summary:
    outcomes: Counter[str] = Counter(label for _, label in result.samples)
    success_latencies = sorted(
        lat for lat, label in result.samples if label == "success"
    )
    if success_latencies:
        avg_ms = (sum(success_latencies) / len(success_latencies)) * 1000.0
        p50_ms = _percentile_ms(success_latencies, 0.50)
        p90_ms = _percentile_ms(success_latencies, 0.90)
        p99_ms = _percentile_ms(success_latencies, 0.99)
    else:
        avg_ms = p50_ms = p90_ms = p99_ms = 0.0
    return Summary(
        backend=result.backend,
        n=result.n,
        wall_s=result.wall_s,
        avg_ms=avg_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        outcomes=outcomes,
    )
