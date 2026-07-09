# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metric_value(metrics: Mapping[str, Any], tag: str) -> float | None:
    value = metrics.get(tag)
    if not isinstance(value, Mapping):
        return None
    raw_value = value.get("value")
    return float(raw_value) if isinstance(raw_value, int | float) else None


def aggregate_metric_value(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, Mapping):
        return None
    for key in ("value", "avg", "sum"):
        raw_value = value.get(key)
        if isinstance(raw_value, int | float):
            return float(raw_value)
    return None


def parse_aiperf_records(artifact_dir: Path) -> dict[str, Any] | None:
    """Recover completion RPS from incremental records if aiperf's finalizer stalls."""

    records_path = artifact_dir / "profile_export.jsonl"
    if not records_path.is_file():
        return None

    starts: list[int] = []
    ends: list[int] = []
    request_latencies: list[float] = []
    ttfts: list[float] = []
    itls: list[float] = []
    isls: list[float] = []
    osls: list[float] = []
    cancelled = 0
    errored = 0
    malformed = 0

    with records_path.open(encoding="utf-8") as records_file:
        for line in records_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            metadata = record.get("metadata") or {}
            if metadata.get("benchmark_phase") != "profiling":
                continue
            if record.get("error") is not None:
                errored += 1
                if metadata.get("was_cancelled"):
                    cancelled += 1
                continue
            if metadata.get("was_cancelled"):
                cancelled += 1
                continue
            start = metadata.get("request_start_ns")
            end = metadata.get("request_end_ns")
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                continue
            starts.append(start)
            ends.append(end)
            metrics = record.get("metrics") or {}
            for values, tag in (
                (request_latencies, "request_latency"),
                (ttfts, "time_to_first_token"),
                (itls, "inter_token_latency"),
                (isls, "input_sequence_length"),
                (osls, "output_sequence_length"),
            ):
                value = metric_value(metrics, tag)
                if value is not None:
                    values.append(value)

    wall_seconds = (max(ends) - min(starts)) / 1_000_000_000 if starts else 0.0
    completed = len(starts)
    return {
        "records_path": str(records_path),
        "completed_profiling_records": completed,
        "cancelled_profiling_records": cancelled,
        "errored_profiling_records": errored,
        "malformed_records": malformed,
        "wall_seconds": wall_seconds,
        "request_throughput_rps": completed / wall_seconds if wall_seconds else None,
        "request_latency_ms": {
            "p50": percentile(request_latencies, 0.50),
            "p95": percentile(request_latencies, 0.95),
            "p99": percentile(request_latencies, 0.99),
        },
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "itl_ms": {
            "p50": percentile(itls, 0.50),
            "p95": percentile(itls, 0.95),
            "p99": percentile(itls, 0.99),
        },
        "isl_tokens": {
            "p50": percentile(isls, 0.50),
            "avg": sum(isls) / len(isls) if isls else None,
        },
        "osl_tokens": {
            "p50": percentile(osls, 0.50),
            "avg": sum(osls) / len(osls) if osls else None,
        },
    }


def parse_aiperf_metrics(artifact_dir: Path) -> dict[str, Any] | None:
    """Persist aggregate and raw-record aiperf metrics beside the arm metadata."""

    summary_path = artifact_dir / "profile_export_aiperf.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            summary = {"parse_error": str(error), "summary_path": str(summary_path)}
        else:

            def metric(tag: str) -> Any:
                return data.get(tag)

            summary = {
                "summary_path": str(summary_path),
                "request_throughput": metric("request_throughput"),
                "inter_token_latency": metric("inter_token_latency"),
                "time_to_first_token": metric("time_to_first_token"),
                "request_latency": metric("request_latency"),
                "input_sequence_length": metric("input_sequence_length"),
                "output_sequence_length": metric("output_sequence_length"),
                "output_token_throughput": metric("output_token_throughput"),
                "request_count": metric("request_count"),
                "good_request_count": metric("good_request_count"),
                "error_request_count": metric("error_request_count"),
                "error_summary": data.get("error_summary"),
                "total_isl": metric("total_isl"),
                "total_osl": metric("total_osl"),
            }

    records = parse_aiperf_records(artifact_dir)
    if summary is None and records is None:
        return None
    return {"summary": summary, "records": records}
