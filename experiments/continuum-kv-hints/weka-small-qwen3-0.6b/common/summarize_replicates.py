# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate repeated WEKA/Qwen KV-hint ablation matrices."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

CASES = (
    "a-no-hints",
    "b-parent-retain",
    "c-root-final-evict",
    "d-combined",
)
METRICS = (
    "request_count",
    "request_error_rate_pct",
    "request_throughput_rps",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "latency_p50_ms",
    "latency_p99_ms",
    "observed_prompt_cache_read_pct",
    "prompt_cache_read_tokens_sum",
    "retain_action_count",
    "evict_action_count",
    "missing_remove_warning_count",
)


def load_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "matrix-summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return {row["case"]: row for row in rows}


def aggregate_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    runs = [load_run(run_dir) for run_dir in run_dirs]
    rows: list[dict[str, Any]] = []
    for case in CASES:
        row: dict[str, Any] = {"case": case, "replicate_count": len(runs)}
        for metric in METRICS:
            values = [float(run[case][metric]) for run in runs]
            for statistic, value in aggregate_metric(values).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def value_range(row: dict[str, Any], metric: str, digits: int) -> str:
    mean = row[f"{metric}_mean"]
    minimum = row[f"{metric}_min"]
    maximum = row[f"{metric}_max"]
    return f"{mean:.{digits}f} [{minimum:.{digits}f}, {maximum:.{digits}f}]"


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Values are mean [minimum, maximum] across replicates.",
        "",
        "| case | requests | req/s | TTFT p50 ms | TTFT p99 ms | latency p50 ms | latency p99 ms | observed cache read | retain actions | evict actions | remove warnings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["case"],
                    value_range(row, "request_count", 1),
                    value_range(row, "request_throughput_rps", 3),
                    value_range(row, "ttft_p50_ms", 1),
                    value_range(row, "ttft_p99_ms", 1),
                    value_range(row, "latency_p50_ms", 1),
                    value_range(row, "latency_p99_ms", 1),
                    value_range(row, "observed_prompt_cache_read_pct", 2) + "%",
                    value_range(row, "retain_action_count", 1),
                    value_range(row, "evict_action_count", 1),
                    value_range(row, "missing_remove_warning_count", 1),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    rows = aggregate_runs(args.run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "replicate-summary.json"
    csv_path = args.output_dir / "replicate-summary.csv"
    markdown_path = args.output_dir / "replicate-summary.md"

    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown(rows), encoding="utf-8")

    print(markdown_path)


if __name__ == "__main__":
    main()
