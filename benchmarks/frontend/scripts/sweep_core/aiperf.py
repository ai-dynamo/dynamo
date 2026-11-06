# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the required AIPerf aggregate artifacts before reporting success."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

from sweep_core.models import RunResult


def validate_artifacts(directory: Path, expected: Optional[int] = None) -> dict:
    """Read aggregate JSON/CSV and cross-check completed request counts."""
    with (directory / "profile_export_aiperf.json").open() as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise ValueError("Benchmark aggregate must be a JSON object")
    counts = []
    for tag in ("request_count", "error_request_count"):
        metric = summary.get(tag, {"avg": 0})
        value = metric["avg"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or int(value) != value
        ):
            raise ValueError(f"Invalid {tag}: {value}")
        counts.append(value)
    successful, errors = counts
    completed = successful + errors
    if completed <= 0 or (expected is not None and completed != expected):
        raise ValueError(
            f"Unexpected completed request count: {completed}; expected {expected}"
        )
    with (directory / "profile_export_aiperf.csv").open(newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    width = None
    count = None
    for row in rows:
        if not row:
            width = None
            continue
        if width is None:
            if len(row) < 2 or row[0] not in {"Metric", "Endpoint"}:
                raise ValueError("Unrecognized benchmark CSV section")
            width = len(row)
        elif len(row) != width:
            raise ValueError("Truncated benchmark CSV row")
        elif row[0] == "Request Count":
            count = float(row[1])
    if count != successful:
        raise ValueError(
            f"CSV request count {count} does not match JSON count {successful}"
        )
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
            candidates = [
                path
                for path in sorted(aiperf_root.rglob(aiperf_json.name))
                if "phases" not in path.relative_to(aiperf_root).parts[:-1]
            ] or candidates
        for candidate in candidates:
            summary = validate_artifacts(candidate.parent, expected)
            parsed = _parse_metrics(summary)
            if not parsed:
                raise ValueError(f"Malformed benchmark metrics: {candidate}")
            if not metrics:
                metrics = parsed
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
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
