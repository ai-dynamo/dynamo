# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create CSV and Markdown reports for the text engine-client benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmarks.multimodal.sweep.experiments.engine_client.validate_text_results import (
    metric,
)

RUNTIME_ORDER = ("vllm-serve", "dynamo-async", "dynamo-sync")
T_CRITICAL_95_N5 = 2.776


def result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "trial": int(path.parents[2].name.removeprefix("trial-")),
                "runtime": path.parents[1].name,
                "request_throughput_rps": metric(document, "request_throughput"),
                "output_throughput_tps": metric(document, "output_token_throughput"),
                "ttft_p50_ms": metric(document, "time_to_first_token", "p50"),
                "ttft_p90_ms": metric(document, "time_to_first_token", "p90"),
                "ttft_p99_ms": metric(document, "time_to_first_token", "p99"),
                "itl_p50_ms": metric(document, "inter_token_latency", "p50"),
                "itl_p90_ms": metric(document, "inter_token_latency", "p90"),
                "itl_p99_ms": metric(document, "inter_token_latency", "p99"),
                "e2e_p50_ms": metric(document, "request_latency", "p50"),
                "e2e_p90_ms": metric(document, "request_latency", "p90"),
                "e2e_p99_ms": metric(document, "request_latency", "p99"),
                "artifact": str(path.parent.relative_to(root)),
                "command": str((path.parent / "command.txt").relative_to(root)),
            }
        )
    return rows


def means_by_runtime(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name, value in row.items():
            if name not in {"trial", "runtime"} and isinstance(value, (int, float)):
                values[str(row["runtime"])][name].append(float(value))
    return {
        runtime: {name: statistics.mean(samples) for name, samples in metrics.items()}
        for runtime, metrics in values.items()
    }


def paired_interval(
    rows: list[dict[str, Any]],
    candidate: str,
    metric_name: str,
) -> tuple[float, float, float]:
    by_trial = {
        (int(row["trial"]), str(row["runtime"])): float(row[metric_name])
        for row in rows
    }
    deltas = [
        (by_trial[(trial, candidate)] / by_trial[(trial, "dynamo-async")] - 1.0) * 100.0
        for trial in range(1, 6)
    ]
    mean = statistics.mean(deltas)
    half_width = T_CRITICAL_95_N5 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean, mean - half_width, mean + half_width


def format_value(
    value: float,
    *,
    best: bool,
    precision: int = 3,
) -> str:
    rendered = f"{value:.{precision}f}"
    return f"**{rendered}**" if best else rendered


def best_runtime(
    means: dict[str, dict[str, float]], metric_name: str, higher: bool
) -> str:
    selector = max if higher else min
    return selector(RUNTIME_ORDER, key=lambda runtime: means[runtime][metric_name])


def markdown_report(root: Path, rows: list[dict[str, Any]]) -> str:
    means = means_by_runtime(rows)
    lines = [
        "# Qwen2.5-1.5B engine-client benchmark",
        "",
        "Five fresh-server trials; concurrency 1; 1,000 measured requests plus 20 warmups; exact ISL 740 / OSL 70.",
        "",
        "## Throughput",
        "",
        "| Runtime | Request throughput (req/s) | Output throughput (tok/s) | Paired request delta vs async (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    best_request = best_runtime(means, "request_throughput_rps", True)
    best_output = best_runtime(means, "output_throughput_tps", True)
    for runtime in RUNTIME_ORDER:
        if runtime == "dynamo-async":
            delta = "baseline"
        else:
            mean, low, high = paired_interval(rows, runtime, "request_throughput_rps")
            delta = f"{mean:+.2f}% [{low:+.2f}, {high:+.2f}]"
        lines.append(
            "| "
            + " | ".join(
                (
                    runtime,
                    format_value(
                        means[runtime]["request_throughput_rps"],
                        best=runtime == best_request,
                    ),
                    format_value(
                        means[runtime]["output_throughput_tps"],
                        best=runtime == best_output,
                    ),
                    delta,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## TTFT and ITL",
            "",
            "| Runtime | TTFT p50 | TTFT p90 | TTFT p99 | ITL p50 | ITL p90 | ITL p99 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    latency_metrics = (
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p90_ms",
        "itl_p99_ms",
    )
    latency_bests = {name: best_runtime(means, name, False) for name in latency_metrics}
    for runtime in RUNTIME_ORDER:
        rendered = [
            format_value(means[runtime][name], best=latency_bests[name] == runtime)
            for name in latency_metrics
        ]
        lines.append(f"| {runtime} | " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## End-to-end latency",
            "",
            "| Runtime | p50 (ms) | p90 (ms) | p99 (ms) | Paired p99 delta vs async (95% CI) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    e2e_metrics = ("e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms")
    e2e_bests = {name: best_runtime(means, name, False) for name in e2e_metrics}
    for runtime in RUNTIME_ORDER:
        if runtime == "dynamo-async":
            delta = "baseline"
        else:
            mean, low, high = paired_interval(rows, runtime, "e2e_p99_ms")
            delta = f"{mean:+.2f}% [{low:+.2f}, {high:+.2f}]"
        rendered = [
            format_value(means[runtime][name], best=e2e_bests[name] == runtime)
            for name in e2e_metrics
        ]
        lines.append(f"| {runtime} | " + " | ".join([*rendered, delta]) + " |")

    lines.extend(
        [
            "",
            "## Artifacts and commands",
            "",
            "| Trial | Runtime | Artifact | Command |",
            "|---:|---|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["trial"], item["runtime"])):
        lines.append(
            f"| {row['trial']} | {row['runtime']} | "
            f"[{row['artifact']}]({row['artifact']}) | "
            f"[command]({row['command']}) |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def report(root: Path) -> tuple[Path, Path]:
    rows = result_rows(root)
    if len(rows) != 15:
        raise ValueError(f"expected 15 audited results, found {len(rows)}")
    csv_path = root / "benchmark.csv"
    markdown_path = root / "benchmark.md"
    write_csv(csv_path, rows)
    markdown_path.write_text(markdown_report(root, rows), encoding="utf-8")
    print(f"benchmark={markdown_path}")
    print(f"csv={csv_path}")
    return markdown_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    report(parse_args().root.resolve())


if __name__ == "__main__":
    main()
