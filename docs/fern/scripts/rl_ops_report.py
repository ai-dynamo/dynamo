#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a checked RL trace join into a request-plane operations report.

The input is the body-free JSONL emitted by rl_trace_join.py. The report uses
only fields in the current Dynamo request-trace schema plus application-owned
framework and policy identity preserved by the join. It exposes field coverage
instead of inventing alert thresholds or treating absent optional telemetry as
zero. Standardized weight-update timing remains outside the request trace.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

SCHEMA = "dynamo.rl.operations-report.v1"
ALLOWED_JOIN_STATUSES = {
    "complete",
    "expected_incomplete",
    "missing_payload",
    "missing_terminal",
    "unexpected_terminal",
}
STRICT_ALLOWED_STATUSES = {"complete", "expected_incomplete"}
METRIC_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "prefill_wait_time_ms",
    "prefill_time_ms",
    "ttft_ms",
    "total_time_ms",
    "avg_itl_ms",
    "kv_hit_rate",
    "kv_transfer_estimated_latency_ms",
    "queue_depth",
)
TOKEN_SUM_FIELDS = {"input_tokens", "output_tokens", "cached_tokens"}
INTEGER_FIELDS = TOKEN_SUM_FIELDS | {"queue_depth"}
WORKER_ROLES = (
    ("prefill", "prefill_worker_id"),
    ("decode", "decode_worker_id"),
)
DATA_BOUNDARIES = [
    "Optional request-trace metrics are reported with coverage and are not imputed as zero.",
    "Application target and observed policy strings prove header correlation, not model-content identity.",
    "The request trace does not emit standardized weight-update phase, duration, result, or served-content events.",
    "Router and backend Prometheus series remain separate aggregate sources and are not joined by this report.",
]


class ReportError(ValueError):
    """Raised when joined input cannot produce a trustworthy report."""


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_joined(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ReportError(f"joined input: {path} does not exist")
        with _open_text(path) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReportError(
                        f"joined input: invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ReportError(
                        f"joined input: row at {path}:{line_number} must be an object"
                    )
                records.append(record)
    if not records:
        raise ReportError("joined input must contain at least one row")
    return records


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _round(value: float) -> float:
    return round(value, 6)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _numeric(value: Any, location: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{location} must be numeric when present")
    if not math.isfinite(value) or value < 0:
        raise ReportError(f"{location} must be finite and nonnegative")
    return value


def _metric_numeric(value: Any, field: str, location: str) -> float | int:
    value = _numeric(value, location)
    if field in INTEGER_FIELDS and not isinstance(value, int):
        raise ReportError(f"{location} must be an integer when present")
    return value


def _metric_summary(values: list[float | int], terminal_rows: int) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    summary: dict[str, Any] = {
        "present": len(values),
        "coverage_percent": (
            _round(100 * len(values) / terminal_rows) if terminal_rows else 0.0
        ),
    }
    if not numeric:
        summary.update({"min": None, "mean": None, "p50": None, "p95": None, "max": None})
        return summary
    summary.update(
        {
            "min": _round(min(numeric)),
            "mean": _round(sum(numeric) / len(numeric)),
            "p50": _round(_percentile(numeric, 0.50)),
            "p95": _round(_percentile(numeric, 0.95)),
            "max": _round(max(numeric)),
        }
    )
    return summary


def _required_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"{location} must be an object")
    return value


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ReportError(f"{location} must be a non-empty string when present")
    return value


def _worker_id(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportError(f"{location} must be a nonnegative integer when present")
    return str(value)


def build_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ReportError("joined input must contain at least one row")

    status_counts: Counter[str] = Counter()
    framework_status_counts: Counter[str] = Counter()
    acceptance_counts: Counter[str] = Counter()
    target_policy_counts: Counter[str] = Counter()
    observed_policy_counts: Counter[str] = Counter()
    policy_match_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    metric_values: dict[str, list[float | int]] = {field: [] for field in METRIC_FIELDS}
    worker_activity: dict[str, dict[str, dict[str, int]]] = {
        role: {} for role, _ in WORKER_ROLES
    }
    terminal_rows = 0

    for index, row in enumerate(rows, start=1):
        location = f"joined row {index}"
        framework = _required_object(row.get("framework"), f"{location}.framework")
        join_status = _optional_string(row.get("join_status"), f"{location}.join_status")
        if join_status not in ALLOWED_JOIN_STATUSES:
            raise ReportError(
                f"{location}.join_status must be one of {sorted(ALLOWED_JOIN_STATUSES)}"
            )
        status_counts[join_status] += 1

        framework_status = _optional_string(
            framework.get("framework_status"), f"{location}.framework.framework_status"
        )
        framework_status_counts[framework_status or "unknown"] += 1
        accepted = framework.get("accepted")
        if accepted is None:
            acceptance_counts["unknown"] += 1
        elif isinstance(accepted, bool):
            acceptance_counts["accepted" if accepted else "rejected"] += 1
        else:
            raise ReportError(f"{location}.framework.accepted must be boolean when present")

        target_policy = _optional_string(
            framework.get("target_policy_version"),
            f"{location}.framework.target_policy_version",
        )
        observed_policy = _optional_string(
            row.get("observed_policy_version"),
            f"{location}.observed_policy_version",
        )
        target_policy_counts[target_policy or "unknown"] += 1
        observed_policy_counts[observed_policy or "unknown"] += 1
        policy_match = row.get("policy_header_matches")
        if policy_match is None:
            policy_match_counts["not_checked"] += 1
        elif isinstance(policy_match, bool):
            policy_match_counts["matched" if policy_match else "mismatched"] += 1
        else:
            raise ReportError(f"{location}.policy_header_matches must be boolean or null")

        request_end = row.get("request_end")
        if request_end is None:
            continue
        request_end = _required_object(request_end, f"{location}.request_end")
        request = _required_object(request_end.get("request"), f"{location}.request_end.request")
        terminal_rows += 1

        for field in METRIC_FIELDS:
            value = request.get(field)
            if value is not None:
                metric_values[field].append(
                    _metric_numeric(
                        value, field, f"{location}.request_end.request.{field}"
                    )
                )

        finish_metadata = request.get("finish_reason_metadata")
        if finish_metadata is not None:
            finish_metadata = _required_object(
                finish_metadata, f"{location}.request_end.request.finish_reason_metadata"
            )
            finish_reason = _optional_string(
                finish_metadata.get("finish_reason"),
                f"{location}.request_end.request.finish_reason_metadata.finish_reason",
            )
            if finish_reason:
                finish_reason_counts[finish_reason] += 1

        worker = request.get("worker")
        if worker is None:
            continue
        worker = _required_object(worker, f"{location}.request_end.request.worker")
        for role, field in WORKER_ROLES:
            identifier = _worker_id(
                worker.get(field), f"{location}.request_end.request.worker.{field}"
            )
            if identifier is None:
                continue
            initial = (
                {"requests": 0, "input_tokens": 0, "cached_tokens": 0}
                if role == "prefill"
                else {"requests": 0, "output_tokens": 0}
            )
            activity = worker_activity[role].setdefault(identifier, initial)
            activity["requests"] += 1
            role_token_fields = (
                ("input_tokens", "cached_tokens")
                if role == "prefill"
                else ("output_tokens",)
            )
            for token_field in role_token_fields:
                value = request.get(token_field)
                if value is not None:
                    activity[token_field] += int(
                        _metric_numeric(
                            value,
                            token_field,
                            f"{location}.request_end.request.{token_field}",
                        )
                    )

    metric_report: dict[str, dict[str, Any]] = {}
    for field in METRIC_FIELDS:
        summary = _metric_summary(metric_values[field], terminal_rows)
        if field in TOKEN_SUM_FIELDS:
            summary["sum"] = sum(metric_values[field])
        metric_report[field] = summary

    return {
        "schema": SCHEMA,
        "rows": {
            "joined": len(rows),
            "terminal": terminal_rows,
            "without_terminal": len(rows) - terminal_rows,
        },
        "join_status": _counter_dict(status_counts),
        "framework_status": _counter_dict(framework_status_counts),
        "acceptance": _counter_dict(acceptance_counts),
        "policy_versions": {
            "target": _counter_dict(target_policy_counts),
            "observed": _counter_dict(observed_policy_counts),
            "header_match": _counter_dict(policy_match_counts),
        },
        "terminal_metrics": metric_report,
        "finish_reasons": _counter_dict(finish_reason_counts),
        "worker_activity": {
            role: {key: activity[key] for key in sorted(activity, key=int)}
            for role, activity in worker_activity.items()
        },
        "data_boundaries": DATA_BOUNDARIES,
    }


def strict_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if report["rows"]["terminal"] == 0:
        findings.append("at least one terminal request is required")
    unexpected = sorted(set(report["join_status"]) - STRICT_ALLOWED_STATUSES)
    if unexpected:
        findings.append(f"strict report rejects join statuses: {', '.join(unexpected)}")
    mismatches = report["policy_versions"]["header_match"].get("mismatched", 0)
    if mismatches:
        findings.append(f"strict report rejects {mismatches} policy-header mismatch(es)")
    return findings


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(load_joined(args.joined_jsonl))
    except ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.report_json:
        _write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict:
        findings = strict_findings(report)
        if findings:
            for finding in findings:
                print(f"ERROR: {finding}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
