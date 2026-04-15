# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse [PERF] timing lines from server logs and produce per-stage breakdown.

Usage:
    python benchmarks/multimodal/sweep/scripts/parse_ttft_breakdown.py \
        --log-dir logs/04-07/h100-vllm-vs-fd-nsys/
"""

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List


# [PERF] component key=value ...
PERF_RE = re.compile(r"\[PERF\]\s+(\S+)\s+(.*)")
TIME_MS_RE = re.compile(r"time_ms=([\d.]+)")
TOTAL_MS_RE = re.compile(r"total_ms=([\d.]+)")
FETCH_MS_RE = re.compile(r"fetch_ms=([\d.]+)")
DECODE_MS_RE = re.compile(r"decode_ms=([\d.]+)")
REGISTER_MS_RE = re.compile(r"register_ms=([\d.]+)")
REQUEST_ID_RE = re.compile(r"request_id=(\S+)")


def parse_log_file(path: Path) -> Dict[str, List[float]]:
    """Parse a single log file for [PERF] lines, returning component -> list of ms values."""
    components: Dict[str, List[float]] = {}

    with open(path) as f:
        for line in f:
            m = PERF_RE.search(line)
            if not m:
                continue

            component = m.group(1)
            rest = m.group(2)

            # Extract timing values
            time_match = TIME_MS_RE.search(rest)
            if time_match:
                components.setdefault(component, []).append(float(time_match.group(1)))

            total_match = TOTAL_MS_RE.search(rest)
            if total_match:
                key = f"{component}.total"
                components.setdefault(key, []).append(float(total_match.group(1)))

            fetch_match = FETCH_MS_RE.search(rest)
            if fetch_match:
                key = f"{component}.fetch"
                components.setdefault(key, []).append(float(fetch_match.group(1)))

            decode_match = DECODE_MS_RE.search(rest)
            if decode_match:
                key = f"{component}.decode"
                components.setdefault(key, []).append(float(decode_match.group(1)))

            register_match = REGISTER_MS_RE.search(rest)
            if register_match:
                key = f"{component}.register"
                components.setdefault(key, []).append(float(register_match.group(1)))

    return components


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile using sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    low = int(idx)
    high = min(low + 1, len(sorted_data) - 1)
    frac = idx - low
    return sorted_data[low] * (1 - frac) + sorted_data[high] * frac


def print_breakdown(
    label: str,
    components: Dict[str, List[float]],
) -> None:
    """Print formatted breakdown table."""
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"  {'Component':<40} {'Count':>6} {'Avg':>8} {'P50':>8} {'P90':>8} {'P99':>8}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for component in sorted(components.keys()):
        values = components[component]
        if not values:
            continue
        avg = statistics.mean(values)
        p50 = percentile(values, 50)
        p90 = percentile(values, 90)
        p99 = percentile(values, 99)
        print(
            f"  {component:<40} {len(values):>6} "
            f"{avg:>7.2f}ms {p50:>7.2f}ms {p90:>7.2f}ms {p99:>7.2f}ms"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse [PERF] timing logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Root directory containing config subdirs with server.log files",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not log_dir.exists():
        print(f"ERROR: {log_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all server.log files
    log_files = sorted(log_dir.rglob("server.log"))
    if not log_files:
        print(f"No server.log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s) in {log_dir}")

    for log_file in log_files:
        # Derive label from path: e.g. dynamo-fd/requestrate4/server.log -> dynamo-fd @ rate=4
        rel = log_file.relative_to(log_dir)
        parts = rel.parts[:-1]  # remove server.log
        label = " / ".join(parts)

        components = parse_log_file(log_file)
        if components:
            print_breakdown(label, components)
        else:
            print(f"\n  {label}: No [PERF] data found")


if __name__ == "__main__":
    main()
