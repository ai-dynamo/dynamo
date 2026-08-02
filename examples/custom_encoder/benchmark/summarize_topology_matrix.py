# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize five-repeat custom-encoder topology benchmark results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    summary = {"repetitions": 5, "topologies": {}}
    for topology in ("inline", "frontend", "worker"):
        paths = sorted(args.result_root.glob(f"rep-*/{topology}/measured.json"))
        if len(paths) != 5:
            raise RuntimeError(f"{topology}: expected 5 results, found {len(paths)}")
        runs = [json.loads(path.read_text()) for path in paths]
        if any(
            run["requests"] != 1000
            or run["successes"] != 1000
            or run["errors"]
            or run["retries"]
            for run in runs
        ):
            raise RuntimeError(f"{topology}: a run failed its request audit")
        wall_times = [float(run["wall_time_s"]) for run in runs]
        throughputs = [float(run["request_throughput"]) for run in runs]
        summary["topologies"][topology] = {
            "wall_time_s": {
                "runs": wall_times,
                "mean": statistics.mean(wall_times),
                "median": statistics.median(wall_times),
                "stdev": statistics.stdev(wall_times),
                "min": min(wall_times),
                "max": max(wall_times),
            },
            "request_throughput": {
                "runs": throughputs,
                "mean_of_runs": statistics.mean(throughputs),
                "from_mean_wall_time": 1000 / statistics.mean(wall_times),
                "stdev": statistics.stdev(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "latency_p95_s_mean": statistics.mean(
                float(run["latency_s"]["p95"]) for run in runs
            ),
        }

    output = args.result_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
