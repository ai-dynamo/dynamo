#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parse aiperf benchmark results and generate a Markdown summary table.
#
# Usage:
#   python3 parse_results.py [--log-dir ./logs]

import argparse
import glob
import json
import os
import re
import sys

ROUTER_LABELS = {
    "mm": "MM",
    "rr": "RR",
    "vllm_processor": "vLLM-P",
    "rust": "Rust",
}
ROUTER_ORDER = ["mm", "rr", "vllm_processor", "rust"]


def find_result_json(artifact_dir: str) -> dict | None:
    """Find and load the aiperf result JSON from an artifact directory."""
    for pattern in ["*.json", "**/*.json"]:
        files = glob.glob(os.path.join(artifact_dir, pattern), recursive=True)
        for f in files:
            try:
                with open(f) as fh:
                    return json.load(fh)
            except Exception:
                continue
    return None


def extract_metrics(data: dict) -> dict | None:
    """
    Extract latency + throughput from aiperf JSON.
    Tries common key name variants.
    Returns dict with keys: min, avg, p50, p90, p99, max, tput
    or None if cannot parse.
    """

    def get(keys, factor=1.0):
        for k in keys:
            if k in data:
                return round(data[k] * factor)
            # nested under "latency" or "stats" sub-dict
            for sub in ("latency", "stats", "results", "metrics"):
                if sub in data and k in data[sub]:
                    return round(data[sub][k] * factor)
        return None

    def get_float(keys, factor=1.0):
        for k in keys:
            if k in data:
                return round(data[k] * factor, 1)
            for sub in ("latency", "stats", "results", "metrics", "throughput"):
                if sub in data and k in data[sub]:
                    return round(data[sub][k] * factor, 1)
        return None

    # Some aiperf versions use seconds, others ms — detect by magnitude
    # If avg > 1000 it's likely already in ms; if avg < 10 it's likely seconds
    raw_avg = get(
        [
            "mean_latency",
            "avg_latency",
            "mean",
            "avg",
            "average_latency",
            "mean_latency_ms",
            "avg_latency_ms",
            "latency_mean_ms",
        ]
    )
    if raw_avg is None:
        return None

    # Auto-detect unit: if looks like seconds (< 10), convert to ms
    factor = 1000.0 if raw_avg < 10 else 1.0

    def ms(keys):
        return get(keys, factor)

    metrics = {
        "min": ms(["min_latency", "min", "latency_min", "min_latency_ms", "p0"]),
        "avg": ms(
            [
                "mean_latency",
                "avg_latency",
                "mean",
                "avg",
                "average_latency",
                "mean_latency_ms",
                "avg_latency_ms",
                "latency_mean_ms",
            ]
        ),
        "p50": ms(
            ["p50_latency", "p50", "percentile_50", "latency_p50", "p50_latency_ms"]
        ),
        "p90": ms(
            ["p90_latency", "p90", "percentile_90", "latency_p90", "p90_latency_ms"]
        ),
        "p99": ms(
            ["p99_latency", "p99", "percentile_99", "latency_p99", "p99_latency_ms"]
        ),
        "max": ms(["max_latency", "max", "latency_max", "max_latency_ms", "p100"]),
        "tput": get_float(
            [
                "throughput",
                "requests_per_second",
                "rps",
                "tput",
                "throughput_rps",
                "req_per_sec",
                "output_throughput",
            ],
            1.0 / factor if factor == 1000.0 else 1.0,
        ),
    }

    if any(v is None for v in metrics.values()):
        # Print raw keys to help debug
        print(
            f"  WARNING: could not extract all metrics. Available keys: {list(data.keys())[:20]}",
            file=sys.stderr,
        )

    return metrics


def fmt_diff(base_val, new_val, is_tput=False):
    """Format comparison cell: +X(+Y%) or -X(-Y%)"""
    if base_val is None or new_val is None:
        return "N/A"
    if is_tput:
        diff = round(new_val - base_val, 1)
        pct = round((new_val - base_val) / base_val * 100) if base_val else 0
    else:
        diff = round(new_val - base_val)
        pct = round((new_val - base_val) / base_val * 100) if base_val else 0
    sign = "+" if diff >= 0 else ""
    pct_sign = "+" if pct >= 0 else ""
    if is_tput:
        return f"{sign}{diff}({pct_sign}{pct}%)"
    return f"{sign}{diff}({pct_sign}{pct}%)"


def generate_table(results: dict, conc: int, mode: str, reuse: str) -> str:
    """Generate a single Markdown table for the given combination."""
    reuse_label = {"90": "~90%", "50": "~50%", "0": "~0%"}[reuse]
    lines = [
        f"### Concurrency = {conc} | {mode.upper()} | reuse {reuse_label}",
        "| router | min | avg | p50 | p90 | p99 | max | tput |",
        "|--------|-----|-----|-----|-----|-----|-----|------|",
    ]

    row_data = {}
    for router in ROUTER_ORDER:
        key = (router, conc, mode, reuse)
        m = results.get(key)
        if m:
            row_data[router] = m

    mm_m = row_data.get("mm")
    rr_m = row_data.get("rr")

    def data_row(router, label, m):
        tput = f"{m['tput']:.1f}" if m.get("tput") is not None else "N/A"
        return (
            f"| {label:<6} | {m.get('min','N/A')} | {m.get('avg','N/A')} | "
            f"{m.get('p50','N/A')} | {m.get('p90','N/A')} | "
            f"{m.get('p99','N/A')} | {m.get('max','N/A')} | {tput} |"
        )

    def diff_row(label, base_m, new_m):
        if base_m is None or new_m is None:
            return f"| {label:<6} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        cols = []
        for k in ("min", "avg", "p50", "p90", "p99", "max"):
            cols.append(fmt_diff(base_m.get(k), new_m.get(k), is_tput=False))
        cols.append(fmt_diff(base_m.get("tput"), new_m.get("tput"), is_tput=True))
        return "| " + label + " | " + " | ".join(cols) + " |"

    for router in ROUTER_ORDER:
        label = ROUTER_LABELS[router]
        m = row_data.get(router)
        if m is None:
            lines.append(f"| {label:<6} | (no data) ||||||| |")
            continue
        lines.append(data_row(router, label, m))
        if router not in ("mm", "rr"):
            lines.append(diff_row("vs MM", mm_m, m))
            lines.append(diff_row("vs RR", rr_m, m))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Parse aiperf results and generate Markdown tables."
    )
    parser.add_argument(
        "--log-dir", default="./logs", help="Path to the benchmark log/artifact dir"
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"ERROR: log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover artifact dirs
    pattern = re.compile(
        r"^(?P<router>.+?)_(?P<workers>\d+)w_(?P<n_req>\d+)req_1img_"
        r"(?P<pool>\d+)pool_(?P<mode>http|datauri)_conc(?P<conc>\d+)$"
    )

    results = {}
    missing = []

    for d in sorted(os.listdir(log_dir)):
        m = pattern.match(d)
        if not m:
            continue
        router = m.group("router")
        n_req = int(m.group("n_req"))
        pool = int(m.group("pool"))
        mode = m.group("mode")
        conc = int(m.group("conc"))

        p90_pool = n_req // 10
        p50_pool = n_req * 6 // 10
        if pool == p90_pool:
            reuse = "90"
        elif pool == p50_pool:
            reuse = "50"
        else:
            reuse = "0"

        full_path = os.path.join(log_dir, d)
        data = find_result_json(full_path)
        if data is None:
            missing.append(d)
            continue

        metrics = extract_metrics(data)
        if metrics is None:
            missing.append(d)
            continue

        key = (router, conc, mode, reuse)
        results[key] = metrics

    if not results:
        print("No results found. Benchmarks may not have completed yet.")
        if missing:
            print(f"Dirs with missing/unparseable JSON: {missing[:5]}")
        sys.exit(0)

    # Collect all combos present
    conc_levels = sorted({k[1] for k in results})
    modes = ["http", "datauri"]
    reuses = ["90", "50", "0"]

    print("# Benchmark Results\n")

    for conc in conc_levels:
        n = conc * 100
        print(f"## Concurrency = {conc} ({n} requests, 2 workers)\n")
        for mode in modes:
            for reuse in reuses:
                table = generate_table(results, conc, mode, reuse)
                print(table)
                print()

    if missing:
        print(
            f"\n> **Note**: {len(missing)} artifact dirs had no parseable results: "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        )


if __name__ == "__main__":
    main()
