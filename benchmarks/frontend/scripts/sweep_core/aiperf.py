# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the required AIPerf aggregate artifacts before reporting success."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

from sweep_core.models import RunResult


def validate_artifacts(directory: Path, expected: Optional[int] = None) -> dict:
    """Read aggregate JSON/CSV and cross-check completed request counts."""
    with (directory / "profile_export_aiperf.json").open() as stream:
        summary = json.load(stream)
    successful = summary["request_count"]["avg"]
    errors = summary.get("error_request_count", {"avg": 0})["avg"]
    completed = successful + errors
    if expected is not None and completed != expected:
        raise ValueError(
            f"Unexpected completed request count: {completed}; expected {expected}"
        )
    with (directory / "profile_export_aiperf.csv").open(newline="") as stream:
        counts = [
            float(row[1])
            for row in csv.reader(stream, strict=True)
            if row and row[0] == "Request Count"
        ]
    if counts != [successful]:
        raise ValueError("Benchmark CSV request count does not match JSON")
    return summary


def _parse_metrics(data: dict) -> dict:
    """Parse aiperf profile_export_aiperf.json."""
    try:
        result = {}
        rt = data.get("request_throughput", {})
        result["req_per_sec"] = rt.get("avg", 0)
        ot = data.get("output_token_throughput", {})
        result["output_tok_per_sec"] = ot.get("avg", 0)
        ttft = data.get("time_to_first_token", data.get("ttft", {}))
        if isinstance(ttft, dict):
            result["ttft_p50_ms"] = ttft.get("p50", 0) or 0
            result["ttft_p99_ms"] = ttft.get("p99", 0) or 0
        itl = data.get("inter_token_latency", data.get("itl", {}))
        if isinstance(itl, dict):
            result["itl_p50_ms"] = itl.get("p50", 0) or 0
            result["itl_p99_ms"] = itl.get("p99", 0) or 0
        bd = data.get("benchmark_duration", 0)
        result["duration_sec"] = bd.get("avg", 0) if isinstance(bd, dict) else (bd or 0)
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def parse_aiperf_into_result(
    result: RunResult, run_dir: Path, expected: Optional[int] = None
) -> None:
    """Parse aiperf results from the run directory into the RunResult."""
    aiperf_root = run_dir / "aiperf"
    aiperf_json = aiperf_root / "profile_export_aiperf.json"
    candidate = aiperf_json
    metrics = {}
    try:
        candidates = [aiperf_json]
        if not aiperf_json.exists():
            candidates = sorted(aiperf_root.glob(f"*/{aiperf_json.name}")) or candidates
        for candidate in candidates:
            summary = validate_artifacts(candidate.parent, expected)
            parsed = _parse_metrics(summary)
            if not parsed:
                raise ValueError(f"Malformed benchmark metrics: {candidate}")
            if not metrics:
                metrics = parsed
    except (OSError, ValueError, KeyError, IndexError, TypeError, csv.Error) as exc:
        result.status = "fail"
        print(f"    Invalid benchmark artifacts at {candidate}: {exc}")
        return
    if metrics:
        result.req_per_sec = metrics.get("req_per_sec", 0)
        result.output_tok_per_sec = metrics.get("output_tok_per_sec", 0)
        result.ttft_p50_ms = metrics.get("ttft_p50_ms", 0)
        result.ttft_p99_ms = metrics.get("ttft_p99_ms", 0)
        result.itl_p50_ms = metrics.get("itl_p50_ms", 0)
        result.itl_p99_ms = metrics.get("itl_p99_ms", 0)
        result.duration_sec = metrics.get("duration_sec", 0)


if __name__ == "__main__":
    requested = sys.argv[2] if len(sys.argv) > 2 else "null"
    validate_artifacts(
        Path(sys.argv[1]), None if requested == "null" else int(requested)
    )
