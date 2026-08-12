# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize three-run inline-versus-frontend custom-encoder results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TOPOLOGIES = ("inline", "frontend")
REPETITIONS = 3
MIN_FRONTEND_TO_INLINE_RATIO = 0.90


def _wire_size_audit(workload: dict) -> dict:
    hidden_size = 1536
    dtype_bytes = 2
    records_by_size: dict[str, dict] = {}
    for record in workload["images"]:
        key = f"{record['width']}x{record['height']}"
        records_by_size.setdefault(key, record)

    compact_total = 0
    dense_total = 0
    by_size = {}
    for entry in workload["image_size_counts"]:
        key = f"{entry['width']}x{entry['height']}"
        requests = int(entry["requests"])
        visual_tokens = int(records_by_size[key]["merged_visual_tokens"])
        decoder_isl = int(workload["observed_decoder_isl_by_image_size"][key])
        compact_bytes = visual_tokens * hidden_size * dtype_bytes
        dense_bytes = decoder_isl * hidden_size * dtype_bytes
        compact_total += requests * compact_bytes
        dense_total += requests * dense_bytes
        by_size[key] = {
            "requests": requests,
            "visual_tokens": visual_tokens,
            "decoder_isl": decoder_isl,
            "compact_bytes_per_request": compact_bytes,
            "dense_bytes_per_request": dense_bytes,
            "reduction_percent": (1 - compact_bytes / dense_bytes) * 100,
        }
    requests = int(workload["requests_per_concurrency"])
    return {
        "contract": "visual rows only; token IDs and mask remain JSON metadata",
        "hidden_size": hidden_size,
        "dtype_bytes": dtype_bytes,
        "by_image_size": by_size,
        "weighted_compact_bytes_per_request": compact_total / requests,
        "weighted_dense_bytes_per_request": dense_total / requests,
        "weighted_reduction_percent": (1 - compact_total / dense_total) * 100,
    }


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
        "repetitions": REPETITIONS,
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
        "handoff_wire_size": _wire_size_audit(workload),
        "topologies": {},
    }
    for topology in TOPOLOGIES:
        paths = sorted(result_root.glob(f"rep-*/{topology}/measured.json"))
        if len(paths) != REPETITIONS:
            raise RuntimeError(
                f"{topology}: expected {REPETITIONS} results, found {len(paths)}"
            )
        runs = [json.loads(path.read_text()) for path in paths]
        if any(
            run["requests"] != 1000
            or run["successes"] != 1000
            or run["errors"]
            or run["retries"]
            or run["completion_tokens"] != {"min": 7, "max": 7, "total": 7000}
            for run in runs
        ):
            raise RuntimeError(f"{topology}: a run failed its request audit")
        request_windows = [float(run["request_window_s"]) for run in runs]
        throughputs = [float(run["request_throughput"]) for run in runs]
        client_walls = [float(run["client_wall_time_s"]) for run in runs]
        summary["topologies"][topology] = {
            "request_window_s": {
                "runs": request_windows,
                "mean": statistics.mean(request_windows),
                "stdev": statistics.stdev(request_windows),
            },
            "request_throughput": {
                "runs": throughputs,
                "mean_of_runs": statistics.mean(throughputs),
                "from_mean_request_window": 1000 / statistics.mean(request_windows),
                "stdev": statistics.stdev(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "client_wall_time_s": {
                "runs": client_walls,
                "mean": statistics.mean(client_walls),
            },
            "latency_p95_s_mean": statistics.mean(
                float(run["latency_s"]["p95"]) for run in runs
            ),
        }

    inline = summary["topologies"]["inline"]["request_throughput"]["mean_of_runs"]
    frontend = summary["topologies"]["frontend"]["request_throughput"]["mean_of_runs"]
    ratio = frontend / inline
    summary["comparison"] = {
        "frontend_to_inline_ratio": ratio,
        "frontend_penalty_percent": (ratio - 1) * 100,
        "minimum_ratio": MIN_FRONTEND_TO_INLINE_RATIO,
        "pass": ratio >= MIN_FRONTEND_TO_INLINE_RATIO,
    }
    (result_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_markdown(summary: dict, output: Path) -> None:
    workload = summary["workload"]
    metadata = summary["metadata"]
    wire = summary["handoff_wire_size"]
    comparison = summary["comparison"]
    lines = [
        "# Inline versus frontend custom encoder",
        "",
        "## Audited workload",
        "",
        f"- Raw text ISL: {workload['text_isl']} tokens, plus one image",
        f"- Decoder ISL by image size: {workload['observed_decoder_isl_by_image_size']}",
        f"- Output tokens: {workload['target_osl']}",
        "- 1,000 measured requests; concurrency 64; 20 warmups; 3 fresh-server repetitions",
        f"- Measured JSONL SHA-256: `{workload['input_sha256']}`",
        f"- Prompt SHA-256: `{workload['prompt_sha256']}`",
        "",
        "## Provenance",
        "",
        f"- Dynamo commit: `{metadata['dynamo_commit']}`",
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
            f"Frontend/inline ratio: **{comparison['frontend_to_inline_ratio']:.3f}** "
            f"({comparison['frontend_penalty_percent']:+.2f}%). Gate >= "
            f"{comparison['minimum_ratio']:.2f}: **{'PASS' if comparison['pass'] else 'FAIL'}**.",
            "",
            "## Handoff size model",
            "",
            f"The compact route transfers {wire['weighted_compact_bytes_per_request']:.0f} "
            "embedding bytes/request on average versus "
            f"{wire['weighted_dense_bytes_per_request']:.0f} for the former dense "
            f"prompt contract, a {wire['weighted_reduction_percent']:.2f}% reduction.",
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
    if not summary["comparison"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
