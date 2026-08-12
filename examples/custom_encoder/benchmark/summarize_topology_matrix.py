# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize five-repeat custom-encoder topology benchmark results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TOPOLOGIES = ("inline", "frontend", "worker")
HISTORICAL_TOTAL_ISL_THROUGHPUT = {
    "inline": 60.3311,
    "frontend": 50.1628,
    "worker": 50.3769,
}


def _percent_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1) * 100


def summarize(result_root: Path) -> dict:
    metadata = json.loads(
        (result_root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    workload = json.loads(
        (result_root / "workloads/measured/workload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "repetitions": 5,
        "metadata": metadata,
        "workload": {
            "text_isl": workload["text_isl"],
            "target_osl": workload["target_osl"],
            "input_sha256": workload["input"]["sha256"],
            "prompt_sha256": workload["prompt_sha256"],
            "observed_decoder_isl_by_image_size": workload[
                "observed_decoder_isl_by_image_size"
            ],
            "image_size_counts": workload["image_size_counts"],
        },
        "topologies": {},
    }
    for topology in TOPOLOGIES:
        paths = sorted(result_root.glob(f"rep-*/{topology}/measured.json"))
        if len(paths) != 5:
            raise RuntimeError(f"{topology}: expected 5 results, found {len(paths)}")
        runs = [json.loads(path.read_text()) for path in paths]
        if any(
            run["requests"] != 1000
            or run["successes"] != 1000
            or run["errors"]
            or run["retries"]
            or run["completion_tokens"]
            != {
                "min": 7,
                "max": 7,
                "total": 7000,
            }
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

    inline = summary["topologies"]["inline"]["request_throughput"]["mean_of_runs"]
    comparisons = {}
    for topology in ("frontend", "worker"):
        separated = summary["topologies"][topology]["request_throughput"][
            "mean_of_runs"
        ]
        comparisons[topology] = {
            "inline_advantage_percent": _percent_change(inline, separated),
            "separated_penalty_percent": _percent_change(separated, inline),
        }
    summary["comparisons"] = comparisons
    summary["historical_total_isl_control"] = {
        "request_throughput": HISTORICAL_TOTAL_ISL_THROUGHPUT,
        "note": "Different workload: text plus visual tokens totaled 644.",
    }

    output = result_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_markdown(summary: dict, output: Path) -> None:
    workload = summary["workload"]
    metadata = summary["metadata"]
    lines = [
        "# Custom encoder topology matrix: 644 raw text tokens plus one image",
        "",
        "## Workload",
        "",
        f"- Text ISL: {workload['text_isl']} raw tokenizer tokens",
        f"- Decoder ISL by image size: {workload['observed_decoder_isl_by_image_size']}",
        f"- Output tokens: {workload['target_osl']}",
        f"- Measured JSONL SHA-256: `{workload['input_sha256']}`",
        f"- Prompt SHA-256: `{workload['prompt_sha256']}`",
        "",
        "## Provenance",
        "",
        f"- Dynamo base commit: `{metadata['dynamo_commit']}`",
        f"- Benchmark patch SHA-256: `{metadata['benchmark_patch_sha256']}`",
        f"- Container: `{metadata['container_image']}`",
        f"- GPU: {metadata['gpu']}",
        f"- Package versions: {metadata['versions']}",
        "",
        "## Results",
        "",
        "| Topology | Runs (req/s) | Mean (req/s) | Sample stdev |",
        "| --- | --- | ---: | ---: |",
    ]
    for topology in TOPOLOGIES:
        throughput = summary["topologies"][topology]["request_throughput"]
        run_text = ", ".join(f"{value:.3f}" for value in throughput["runs"])
        lines.append(
            f"| {topology} | {run_text} | {throughput['mean_of_runs']:.3f} | "
            f"{throughput['stdev']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Relative differences",
            "",
            "| Separated topology | Inline advantage | Separated penalty |",
            "| --- | ---: | ---: |",
        ]
    )
    for topology in ("frontend", "worker"):
        comparison = summary["comparisons"][topology]
        lines.append(
            f"| {topology} | {comparison['inline_advantage_percent']:+.2f}% | "
            f"{comparison['separated_penalty_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                "The historical 60.33/50.16/50.38 req/s control used a different "
                "contract: text plus visual tokens totaled 644. It is retained for "
                "context and is not directly pooled with this result."
            ),
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    summary = summarize(args.result_root)
    write_markdown(summary, args.result_root / "report.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
