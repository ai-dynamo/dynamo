# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Artifact writers for sweep results.

Produces CSV, markdown summary, and sweep_config.json -- the contract
consumed by downstream analysis tools (analyze_sweep.py, sweep_data.py).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List

from sweep_core.models import RunResult, SweepConfig


def write_csv(results: List[RunResult], csv_path: Path, config: SweepConfig) -> None:
    """Write incremental CSV results file (called after each run)."""
    fieldnames = [
        "run_id",
        "backend",
        "tokenizer",
        "concurrency",
        "isl",
        "osl",
        "workers",
        "speedup_ratio",
        "status",
        "req_per_sec",
        "output_tok_per_sec",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "duration_sec",
        "run_dir",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            spec = r.run_spec
            row = {
                "run_id": spec.run_id,
                "backend": spec.deploy.backend,
                "tokenizer": spec.deploy.tokenizer,
                "concurrency": spec.aiperf.concurrency,
                "isl": spec.aiperf.isl,
                "osl": spec.aiperf.osl,
                "workers": spec.deploy.workers,
                "speedup_ratio": config.speedup_ratio,
                "status": r.status,
                "req_per_sec": f"{r.req_per_sec:.2f}"
                if r.req_per_sec is not None
                else "",
                "output_tok_per_sec": f"{r.output_tok_per_sec:.1f}"
                if r.output_tok_per_sec is not None
                else "",
                "ttft_p50_ms": f"{r.ttft_p50_ms:.1f}"
                if r.ttft_p50_ms is not None
                else "",
                "ttft_p99_ms": f"{r.ttft_p99_ms:.1f}"
                if r.ttft_p99_ms is not None
                else "",
                "itl_p50_ms": f"{r.itl_p50_ms:.1f}" if r.itl_p50_ms is not None else "",
                "itl_p99_ms": f"{r.itl_p99_ms:.1f}" if r.itl_p99_ms is not None else "",
                "duration_sec": f"{r.duration_sec:.1f}"
                if r.duration_sec is not None
                else "",
                "run_dir": r.run_dir,
            }
            writer.writerow(row)


def write_summary(results: List[RunResult], summary_path: Path) -> None:
    """Write markdown summary table."""
    lines = ["# Sweep Summary\n"]
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(
        "| Run ID | Req/s | Tok/s | TTFT p50 | TTFT p99 | ITL p50 | Duration | Status |"
    )
    lines.append(
        "|--------|------:|------:|---------:|---------:|--------:|---------:|--------|"
    )

    for r in results:
        rps = f"{r.req_per_sec:.1f}" if r.req_per_sec is not None else "-"
        tps = f"{r.output_tok_per_sec:.0f}" if r.output_tok_per_sec is not None else "-"
        tp50 = f"{r.ttft_p50_ms:.1f}ms" if r.ttft_p50_ms is not None else "-"
        tp99 = f"{r.ttft_p99_ms:.1f}ms" if r.ttft_p99_ms is not None else "-"
        ip50 = f"{r.itl_p50_ms:.1f}ms" if r.itl_p50_ms is not None else "-"
        dur = f"{r.duration_sec:.0f}s" if r.duration_sec is not None else "-"
        lines.append(
            f"| {r.run_spec.run_id} | {rps} | {tps} | {tp50} | {tp99} | {ip50} | {dur} | {r.status} |"
        )

    lines.append("")
    ok = sum(1 for r in results if r.status == "ok")
    fail = sum(1 for r in results if r.status == "fail")
    skip = sum(1 for r in results if r.status == "skipped")
    lines.append(
        f"**Totals:** {ok} passed, {fail} failed, {skip} skipped out of {len(results)}"
    )

    summary_path.write_text("\n".join(lines) + "\n")


def write_sweep_config(
    config: SweepConfig, output_dir: Path, total_runs: int = 0
) -> None:
    """Write sweep_config.json for downstream consumers."""
    config_path = output_dir / "sweep_config.json"
    config_data = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "mode": config.mode,
        "model": config.model,
        "model_name": config.model_name,
        "backend": config.backend,
        "backends": config.backend,
        "tokenizers": ",".join(config.tokenizers),
        "isl_list": ",".join(str(i) for i in config.isls),
        "concurrency_list": ",".join(str(c) for c in config.concurrencies),
        "benchmark_duration": config.benchmark_duration or "N/A",
        "osl": config.osl,
        "speedup_ratio": config.speedup_ratio,
        "output_dir": config.output_dir,
        "total_runs": total_runs,
        "isolation_policy": config.isolation_policy,
    }
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")


def print_results_table(results: List[RunResult]) -> None:
    """Print a compact results table to stdout."""
    print(f"\n{'=' * 90}")
    print(
        f"  {'Run ID':<30} {'Req/s':>8} {'Tok/s':>8} {'TTFT p50':>10} {'TTFT p99':>10} {'Status':>8}"
    )
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
    for r in results:
        rps = f"{r.req_per_sec:.1f}" if r.req_per_sec is not None else "N/A"
        tps = (
            f"{r.output_tok_per_sec:.0f}" if r.output_tok_per_sec is not None else "N/A"
        )
        tp50 = f"{r.ttft_p50_ms:.1f}ms" if r.ttft_p50_ms is not None else "N/A"
        tp99 = f"{r.ttft_p99_ms:.1f}ms" if r.ttft_p99_ms is not None else "N/A"
        print(
            f"  {r.run_spec.run_id:<30} {rps:>8} {tps:>8} {tp50:>10} {tp99:>10} {r.status:>8}"
        )
    print(f"{'=' * 90}")
