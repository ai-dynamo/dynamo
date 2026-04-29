#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate AIPerf artifacts from run_sweep.sh into a markdown RESULTS.md.

Walks `<results-root>/<topology>/<workload>_c<concurrency>_isl-<prompt-label>/`
directories produced by `run_sweep.sh`, reads each `profile_export_aiperf.json`,
and emits one markdown section per (workload, prompt_label) with rows for each
(concurrency, topology) cell.

Reuses metric names AIPerf already reports (no new stats). For deeper
regression analysis use `benchmarks/frontend/scripts/analysis/frontend_perf_analysis.py compare`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CELL_RE = re.compile(
    r"^(?P<workload>[a-z]+)_c(?P<concurrency>\d+)_isl-(?P<isl>[a-z]+)$"
)
METRIC_KEYS = [
    ("time_to_first_token", "TTFT (ms)"),
    ("inter_token_latency", "ITL (ms)"),
    ("request_latency", "E2E (ms)"),
    ("output_token_throughput", "Out Tok/s"),
    ("request_throughput", "Req/s"),
]
PERCENTILES = ["avg", "p50", "p90", "p99"]


def load_metrics(artifact_dir: Path) -> dict | None:
    f = artifact_dir / "profile_export_aiperf.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())


def fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}"


def cell_value(metrics: dict, key: str, percentile: str) -> float | None:
    block = metrics.get(key)
    if not isinstance(block, dict):
        return None
    return block.get(percentile)


def walk_results(results_root: Path) -> dict[tuple[str, str, int, str], dict]:
    out: dict[tuple[str, str, int, str], dict] = {}
    if not results_root.is_dir():
        return out
    for topology_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in topology_dir.iterdir() if p.is_dir()):
            m = CELL_RE.match(cell_dir.name)
            if not m:
                continue
            metrics = load_metrics(cell_dir)
            if metrics is None:
                continue
            key = (
                m["workload"],
                m["isl"],
                int(m["concurrency"]),
                topology_dir.name,
            )
            out[key] = metrics
    return out


def render(results_root: Path) -> str:
    table = walk_results(results_root)
    if not table:
        return "# Qwen3-Omni Benchmark Results\n\nNo AIPerf artifacts found.\n"

    topologies = sorted({key[3] for key in table})
    workloads = sorted({key[0] for key in table})
    isls = sorted({key[1] for key in table})

    lines: list[str] = ["# Qwen3-Omni Benchmark Results (DYN-2581)\n"]
    lines.append(
        "Per-cell AIPerf results from `run_sweep.sh`. "
        f"Topologies compared: {', '.join(topologies)}.\n"
    )

    for workload in workloads:
        for isl in isls:
            lines.append(f"## {workload} — prompt={isl}\n")
            for metric_key, metric_label in METRIC_KEYS:
                lines.append(f"### {metric_label}\n")
                header_parts = ["Concurrency"]
                for topology in topologies:
                    for percentile in PERCENTILES:
                        header_parts.append(f"{topology} {percentile}")
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("|" + "|".join("---" for _ in header_parts) + "|")
                concs = sorted(
                    {key[2] for key in table if key[0] == workload and key[1] == isl}
                )
                for c in concs:
                    row = [str(c)]
                    for topology in topologies:
                        metrics = table.get((workload, isl, c, topology), {})
                        for percentile in PERCENTILES:
                            row.append(fmt(cell_value(metrics, metric_key, percentile)))
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
            lines.append("")

    lines.append("## Summary\n")
    lines.append(
        "_Recommendation paragraph: which (workload, concurrency, prompt-len) "
        "regimes favor disagg, where vllm-serve still wins. Fill in once the "
        "full sweep completes._\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory produced by run_sweep.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "RESULTS.md",
        help="Markdown file to write",
    )
    args = parser.parse_args()

    md = render(args.results_root)
    args.output.write_text(md)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
