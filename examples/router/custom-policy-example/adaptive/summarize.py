# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize paired, complete simulation cohorts; requires only the standard library."""

import argparse
import csv
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    args = parser.parse_args()
    groups = defaultdict(dict)
    with open(args.csv, newline="") as source:
        for row in csv.DictReader(source):
            if row["phase"] != "all":
                continue
            if int(row["completed"]) != 10_000:
                raise ValueError("incomplete cohort")
            key = row["scenario"], row["policy"]
            if row["seed"] in groups[key]:
                raise ValueError("duplicate seed")
            groups[key][row["seed"]] = row
    if not groups:
        raise ValueError("no complete cohorts found")
    print(
        "| Scenario | Policy | Mean TTFT (s) | P99 TTFT (s) | TTFT delta vs 0.5 | Paired delta range | Throughput delta |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for (scenario, policy), rows in sorted(groups.items()):
        baseline = groups[scenario, "static-0.5"]
        if rows.keys() != baseline.keys() or rows.keys() != {
            str(i) for i in range(1, 6)
        }:
            raise ValueError("unpaired seeds")
        means = {
            key: statistics.mean(float(r[key]) for r in rows.values())
            for key in ["mean_ttft_s", "p99_ttft_s"]
        }
        deltas = {
            key: [
                100 * (float(r[key]) / float(baseline[seed][key]) - 1)
                for seed, r in rows.items()
            ]
            for key in ["mean_ttft_s", "cohort_req_s"]
        }
        ttft = deltas["mean_ttft_s"]
        print(
            f"| {scenario} | {policy} | {means['mean_ttft_s']:.3f} | {means['p99_ttft_s']:.3f} | {statistics.mean(ttft):+.1f}% | [{min(ttft):+.1f}%, {max(ttft):+.1f}%] | {statistics.mean(deltas['cohort_req_s']):+.2f}% |"
        )


if __name__ == "__main__":
    main()
