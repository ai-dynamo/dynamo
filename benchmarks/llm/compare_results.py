#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare aiperf benchmark results between a baseline and candidate deployment.

Reads profile_export_aiperf.json from both deployment artifact directories,
compares key metrics (TTFT, ITL, throughput, request latency) across
concurrency levels, and flags regressions exceeding a configurable threshold.

Usage:
    python compare_results.py \\
        --baseline artifacts/sglang \\
        --candidate artifacts/dynamo_sglang \\
        --threshold 5.0
"""

import argparse
import json
import os
import re
import sys


# Metrics where LOWER is better (latencies)
LOWER_IS_BETTER = {
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
}
# Metrics where HIGHER is better (throughput)
HIGHER_IS_BETTER = {
    "output_token_throughput",
    "request_throughput",
}

# Which stat to compare for each metric
METRIC_STAT = {
    "time_to_first_token": "avg",
    "inter_token_latency": "avg",
    "request_latency": "avg",
    "output_token_throughput": "avg",
    "request_throughput": "avg",
}

# Human-friendly names and units
METRIC_DISPLAY = {
    "time_to_first_token": ("TTFT", "ms"),
    "inter_token_latency": ("ITL", "ms"),
    "request_latency": ("Request Latency", "ms"),
    "output_token_throughput": ("Output Throughput", "tok/s"),
    "request_throughput": ("Request Throughput", "req/s"),
}


def find_aiperf_results(artifact_dir: str) -> dict[int, dict]:
    """Find and load aiperf results keyed by concurrency level.

    Expected directory structure:
        artifact_dir/concurrency{N}/profile_export_aiperf.json
    """
    results = {}
    if not os.path.isdir(artifact_dir):
        print(f"WARNING: directory not found: {artifact_dir}")
        return results

    for entry in sorted(os.listdir(artifact_dir)):
        match = re.match(r"concurrency(\d+)", entry)
        if not match:
            continue
        concurrency = int(match.group(1))
        entry_path = os.path.join(artifact_dir, entry)

        # Walk to find profile_export_aiperf.json (aiperf nests it)
        json_path = None
        for root, _, files in os.walk(entry_path):
            if "profile_export_aiperf.json" in files:
                json_path = os.path.join(root, "profile_export_aiperf.json")
                break

        if json_path is None:
            print(f"WARNING: No profile_export_aiperf.json in {entry_path}")
            continue

        with open(json_path) as f:
            results[concurrency] = json.load(f)

    return results


def get_metric_value(data: dict, metric: str, stat: str) -> float | None:
    """Extract a metric value from aiperf JSON data."""
    metric_data = data.get(metric)
    if metric_data is None:
        return None
    value = metric_data.get(stat)
    if value is None:
        return None
    return float(value)


def compute_delta_pct(
    baseline_val: float, candidate_val: float, metric: str
) -> float:
    """Compute percentage delta (positive = candidate is WORSE).

    For lower-is-better metrics: positive means candidate is slower.
    For higher-is-better metrics: positive means candidate has less throughput.
    """
    if baseline_val == 0:
        return 0.0

    if metric in LOWER_IS_BETTER:
        # e.g., latency: candidate 110ms vs baseline 100ms => +10% (worse)
        return ((candidate_val - baseline_val) / baseline_val) * 100.0
    else:
        # e.g., throughput: candidate 90 vs baseline 100 => +10% (worse)
        return ((baseline_val - candidate_val) / baseline_val) * 100.0


def compare_and_report(
    baseline_results: dict[int, dict],
    candidate_results: dict[int, dict],
    baseline_name: str,
    candidate_name: str,
    threshold: float,
) -> bool:
    """Compare results and print a formatted report.

    Returns True if all metrics pass (no regression > threshold).
    """
    all_concurrencies = sorted(
        set(baseline_results.keys()) & set(candidate_results.keys())
    )

    if not all_concurrencies:
        print("ERROR: No matching concurrency levels found between baseline and candidate.")
        return False

    all_metrics = list(LOWER_IS_BETTER | HIGHER_IS_BETTER)
    all_pass = True
    regressions = []

    # Print header
    print(f"\n{'='*90}")
    print(f"  Comparison: {baseline_name} (baseline) vs {candidate_name} (candidate)")
    print(f"  Threshold: {threshold}% (positive delta = candidate is worse)")
    print(f"{'='*90}")

    for metric in all_metrics:
        display_name, unit = METRIC_DISPLAY.get(metric, (metric, ""))
        stat = METRIC_STAT.get(metric, "avg")

        print(f"\n  {display_name} ({unit}) — comparing '{stat}':")
        print(f"  {'Concurrency':>12}  {baseline_name:>14}  {candidate_name:>14}  {'Delta %':>10}  {'Status':>8}")
        print(f"  {'-'*12:>12}  {'-'*14:>14}  {'-'*14:>14}  {'-'*10:>10}  {'-'*8:>8}")

        for concurrency in all_concurrencies:
            base_val = get_metric_value(baseline_results[concurrency], metric, stat)
            cand_val = get_metric_value(candidate_results[concurrency], metric, stat)

            if base_val is None or cand_val is None:
                print(f"  {concurrency:>12}  {'N/A':>14}  {'N/A':>14}  {'N/A':>10}  {'SKIP':>8}")
                continue

            delta = compute_delta_pct(base_val, cand_val, metric)
            status = "PASS" if delta <= threshold else "FAIL"
            if delta > threshold:
                all_pass = False
                regressions.append((display_name, concurrency, delta))

            print(
                f"  {concurrency:>12}  {base_val:>14.2f}  {cand_val:>14.2f}  {delta:>+9.1f}%  {status:>8}"
            )

    # Summary
    print(f"\n{'='*90}")
    if all_pass:
        print(f"  RESULT: PASS — {candidate_name} is within {threshold}% of {baseline_name} on all metrics.")
    else:
        print(f"  RESULT: FAIL — {candidate_name} has regressions exceeding {threshold}%:")
        for name, conc, delta in regressions:
            print(f"    - {name} @ concurrency {conc}: {delta:+.1f}%")
    print(f"{'='*90}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Compare aiperf results between baseline and candidate deployments."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline deployment artifacts directory",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to candidate deployment artifacts directory",
    )
    parser.add_argument(
        "--baseline-name",
        default="baseline",
        help="Display name for baseline (default: baseline)",
    )
    parser.add_argument(
        "--candidate-name",
        default="candidate",
        help="Display name for candidate (default: candidate)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Percentage threshold for flagging regressions (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON comparison results",
    )
    args = parser.parse_args()

    baseline_results = find_aiperf_results(args.baseline)
    candidate_results = find_aiperf_results(args.candidate)

    if not baseline_results:
        print(f"ERROR: No aiperf results found in baseline directory: {args.baseline}")
        sys.exit(1)
    if not candidate_results:
        print(f"ERROR: No aiperf results found in candidate directory: {args.candidate}")
        sys.exit(1)

    print(f"Found {len(baseline_results)} concurrency levels in baseline: {sorted(baseline_results.keys())}")
    print(f"Found {len(candidate_results)} concurrency levels in candidate: {sorted(candidate_results.keys())}")

    passed = compare_and_report(
        baseline_results,
        candidate_results,
        args.baseline_name,
        args.candidate_name,
        args.threshold,
    )

    # Optionally write JSON output
    if args.output:
        all_concurrencies = sorted(
            set(baseline_results.keys()) & set(candidate_results.keys())
        )
        comparison_data = []
        for concurrency in all_concurrencies:
            row: dict = {"concurrency": concurrency}
            for metric in list(LOWER_IS_BETTER | HIGHER_IS_BETTER):
                stat = METRIC_STAT.get(metric, "avg")
                base_val = get_metric_value(
                    baseline_results[concurrency], metric, stat
                )
                cand_val = get_metric_value(
                    candidate_results[concurrency], metric, stat
                )
                if base_val is not None and cand_val is not None:
                    delta = compute_delta_pct(base_val, cand_val, metric)
                    row[metric] = {
                        "baseline": base_val,
                        "candidate": cand_val,
                        "delta_pct": round(delta, 2),
                        "pass": delta <= args.threshold,
                    }
            comparison_data.append(row)

        with open(args.output, "w") as f:
            json.dump(
                {
                    "baseline_name": args.baseline_name,
                    "candidate_name": args.candidate_name,
                    "threshold_pct": args.threshold,
                    "overall_pass": passed,
                    "results": comparison_data,
                },
                f,
                indent=2,
            )
        print(f"Comparison JSON written to: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
