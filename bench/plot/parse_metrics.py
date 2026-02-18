# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse Prometheus exposition-format metric scrapes and compute per-concurrency
deltas for cumulative counters/histograms.

Primary use: extract ``dynamo_frontend_tokenizer_latency_ms`` (tokenize and
detokenize) for scenario 2 (Dynamo Rust frontend).
"""

from __future__ import annotations

import re
from pathlib import Path

# Regex that matches a Prometheus metric line (ignoring comments/TYPE/HELP).
# Example:  dynamo_frontend_tokenizer_latency_ms_sum{operation="tokenize"} 116.17
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[^\s]+)$"
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into a flat dict.

    Keys are ``metric_name`` or ``metric_name{label="value",...}`` and values
    are floats.  Only simple gauge/counter/summary _sum/_count lines and
    histogram _sum/_count lines are captured (bucket lines are skipped for
    brevity, though they parse fine).

    Returns
    -------
    dict[str, float]
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if m:
            full_key = m.group("name")
            if m.group("labels"):
                full_key += "{" + m.group("labels") + "}"
            try:
                result[full_key] = float(m.group("value"))
            except ValueError:
                pass
    return result


def load_concurrency_metrics(
    metrics_dir: Path,
    concurrencies: list[int],
) -> dict[int, dict[str, float]]:
    """Load Prometheus scrapes for each concurrency level.

    Parameters
    ----------
    metrics_dir : Path
        Directory containing ``baseline_metrics.txt``,
        ``concurrency_N_metrics.txt``, and ``final_metrics.txt``.
    concurrencies : list[int]
        Ordered list of concurrency levels (e.g. [1, 2, 4, ..., 1024]).

    Returns
    -------
    dict mapping concurrency (int) -> parsed metric dict.
    The special key 0 is used for the baseline scrape.
    """
    result: dict[int, dict[str, float]] = {}

    baseline = metrics_dir / "baseline_metrics.txt"
    if baseline.exists():
        result[0] = parse_prometheus_text(baseline.read_text(errors="replace"))

    for c in concurrencies:
        f = metrics_dir / f"concurrency_{c}_metrics.txt"
        if f.exists():
            result[c] = parse_prometheus_text(f.read_text(errors="replace"))

    return result


def compute_tokenizer_latency_deltas(
    prom_by_concurrency: dict[int, dict[str, float]],
    concurrencies: list[int],
) -> dict[str, dict[int, float]]:
    """Compute average tokenizer latency per concurrency level from deltas.

    Returns
    -------
    dict with keys "tokenize" and "detokenize", each mapping
    concurrency -> avg latency in ms.
    """
    results: dict[str, dict[int, float]] = {"tokenize": {}, "detokenize": {}}

    # Build ordered list: [baseline(0), c1, c2, ...]
    ordered = [0] + list(concurrencies)
    # Filter to those we actually have data for
    ordered = [c for c in ordered if c in prom_by_concurrency]

    for op in ("tokenize", "detokenize"):
        sum_key = f'dynamo_frontend_tokenizer_latency_ms_sum{{operation="{op}"}}'
        count_key = f'dynamo_frontend_tokenizer_latency_ms_count{{operation="{op}"}}'

        for i in range(1, len(ordered)):
            c_prev = ordered[i - 1]
            c_curr = ordered[i]

            prev = prom_by_concurrency[c_prev]
            curr = prom_by_concurrency[c_curr]

            if sum_key not in curr or count_key not in curr:
                continue
            if sum_key not in prev or count_key not in prev:
                continue

            delta_sum = curr[sum_key] - prev[sum_key]
            delta_count = curr[count_key] - prev[count_key]

            if delta_count > 0:
                results[op][c_curr] = delta_sum / delta_count

    return results
