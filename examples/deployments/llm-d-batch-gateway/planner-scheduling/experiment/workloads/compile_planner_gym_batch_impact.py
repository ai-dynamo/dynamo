#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile repeated Planner Gym Batch-impact runs into one campaign report.

The input directories are immutable top-level raw runs produced by
``run_planner_gym_batch_impact.py``. The compiler never mutates those inputs.
It validates their one-cell Planner Gym result, reads AIPerf's native error
counters, validates nested Batch harness evidence, and publishes a new JSON and
Markdown analysis directory atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

ARMS = ("online-only", "stock", "planner-native")
ONLINE_METRICS = (
    "goodput_rps",
    "good_rate",
    "request_throughput_rps",
    "completed_requests",
    "duration_s",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_ms",
    "p99_e2e_ms",
    "error_count",
    "error_rate",
)
BATCH_METRICS = (
    "total",
    "completed",
    "failed",
    "duration_seconds",
    "average_completion_rate_rps",
    "peak_interval_completion_rate_rps",
    "overlap_seconds",
    "overlap_fraction_of_online",
    "completions_during_online",
    "completion_rate_during_online_rps",
)
DECISION_ONLINE_METRICS = (
    "goodput_rps",
    "good_rate",
    "request_throughput_rps",
    "p99_ttft_ms",
    "p99_itl_ms",
    "p99_e2e_ms",
    "error_rate",
)
DECISION_BATCH_METRICS = (
    "average_completion_rate_rps",
    "completion_rate_during_online_rps",
)
LOWER_IS_BETTER = {
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_ms",
    "p99_e2e_ms",
    "error_count",
    "error_rate",
    "failed",
}
MAX_PROGRESS_RECORDS = 100_000
DEFAULT_MINIMUM_BATCH_OVERLAP_FRACTION = 0.95
FRONTEND_POD_PREFIX = "qwen3-0-6b-batch-frontend-"
WORKER_POD_PREFIX = "qwen3-0-6b-batch-worker-"
PLANNER_POD_PREFIX = "qwen3-0-6b-batch-planner-"
ASYNC_POD_PREFIX = "async-dispatch-llm-d-async-"
FRONTEND_REQUESTS_STARTED = "dynamo_frontend_requests_started_total"
ASYNC_BROKER_BACKLOG = "llm_d_async_async_broker_backlog"
ASYNC_BACKLOG_SOURCE_AVAILABLE = "llm_d_async_async_broker_backlog_source_available"
ASYNC_QUEUE_DEPTH = "llm_d_async_async_queue_depth"
ASYNC_DRAIN_LIMIT_RPS = "llm_d_async_async_drain_limit_rps"
ASYNC_DRAIN_LIMIT_LEASE_VALID = "llm_d_async_async_drain_limit_lease_valid"
ASYNC_DRAIN_LIMIT_VALID_UNTIL = "llm_d_async_async_drain_limit_valid_until_seconds"
PHASE_TRANSITION_MARGIN_SECONDS = 15.0
MINIMUM_STABLE_PHASE_SAMPLES = 3
LOW_LOAD_RPS_RANGE = (4.0, 6.0)
HIGH_LOAD_RPS_RANGE = (7.0, 9.0)
LOW_DRAIN_CAP_RANGE = (0.25, 1.0)
HIGH_DRAIN_CAP_MAXIMUM = 0.25
PLANNER_DECISION_PATTERN = re.compile(
    r"Batch scheduling decision: "
    r"pipeline_action=(?P<pipeline_action>\S+) "
    r"pool_id=(?P<pool_id>\S+) "
    r"replica_floor=(?P<replica_floor>\S+) "
    r"max_admission_rps=(?P<max_admission_rps>\S+) "
    r"decision_id=(?P<decision_id>\S+) "
    r"valid_until_s=(?P<valid_until_s>\S+)"
)
PROMETHEUS_SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+[^\s]+)?$"
)
PROMETHEUS_LABEL_PATTERN = re.compile(
    r'\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)="' r'(?P<value>(?:\\.|[^"\\])*)"\s*(?:,|$)'
)


class CampaignCompileError(RuntimeError):
    """The campaign cannot be compiled safely."""


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CampaignCompileError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignCompileError(f"{path} does not contain a JSON object")
    return value


def _read_json_array(path: Path) -> list[Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CampaignCompileError(f"cannot read {path}: {error}") from error
    if not isinstance(value, list):
        raise CampaignCompileError(f"{path} does not contain a JSON array")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise CampaignCompileError(f"cannot read {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(records) >= MAX_PROGRESS_RECORDS:
                raise CampaignCompileError(
                    f"{path} exceeds the {MAX_PROGRESS_RECORDS}-record limit"
                )
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError) as error:
                raise CampaignCompileError(
                    f"{path} line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise CampaignCompileError(
                    f"{path} line {line_number} is not an object"
                )
            records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _parse_time(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise CampaignCompileError(f"{label} is not a timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise CampaignCompileError(f"{label} is not an ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise CampaignCompileError(f"{label} does not include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignCompileError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignCompileError(f"{label} is not finite")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0:
        raise CampaignCompileError(f"{label} is negative")
    return result


def _aiperf_stat(document: dict[str, Any], tag: str) -> float | None:
    value = document.get(tag)
    if isinstance(value, dict):
        value = value.get("avg")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_summary(values: Iterable[float]) -> dict[str, Any]:
    samples = list(values)
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "sample_stdev": None,
            "cv_percent": None,
        }
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) >= 2 else None
    if stdev is None:
        cv = None
    elif mean == 0:
        cv = 0.0 if stdev == 0 else None
    else:
        cv = abs(stdev / mean) * 100.0
    return {
        "count": len(samples),
        "mean": mean,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "sample_stdev": stdev,
        "cv_percent": cv,
    }


def _add_invalid(record: dict[str, Any], message: str) -> None:
    if message not in record["invalid_reasons"]:
        record["invalid_reasons"].append(message)
    record["valid"] = False


def _validate_outer_checksum_manifest(
    run_directory: Path, record: dict[str, Any]
) -> None:
    """Verify every source consumed by the compiler against the outer manifest."""
    manifest_path = run_directory / "artifact-checksums.json"
    if not manifest_path.is_file():
        raise CampaignCompileError("missing artifact-checksums.json")
    manifest = _read_json_object(manifest_path)
    for relative_path, provenance in record["source_files"].items():
        expected = manifest.get(relative_path)
        if not isinstance(expected, str):
            raise CampaignCompileError(
                f"artifact checksum manifest omits {relative_path}"
            )
        if expected != provenance["sha256"]:
            raise CampaignCompileError(
                f"artifact checksum manifest disagrees for {relative_path}"
            )
    record["provenance"]["outer_artifact_manifest"] = {
        **_file_provenance(manifest_path),
        "verified_source_file_count": len(record["source_files"]),
    }


def _extract_online(
    run_directory: Path, record: dict[str, Any]
) -> tuple[dt.datetime, dt.datetime]:
    results_path = run_directory / "planner-gym" / "results.json"
    if not results_path.is_file():
        raise CampaignCompileError("missing planner-gym/results.json")
    report = _read_json_object(results_path)
    record["source_files"]["planner-gym/results.json"] = _file_provenance(results_path)
    report_summary = report.get("summary")
    if not isinstance(report_summary, dict):
        raise CampaignCompileError("Planner Gym report summary is missing")
    if report_summary.get("status") != "ok":
        raise CampaignCompileError("Planner Gym report status is not ok")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise CampaignCompileError("Planner Gym report must contain exactly one result")
    result = results[0]
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise CampaignCompileError("Planner Gym result status is not ok")
    metrics = result.get("metrics")
    raw_metrics = result.get("raw_metrics")
    if not isinstance(metrics, dict) or not isinstance(raw_metrics, dict):
        raise CampaignCompileError("Planner Gym metrics are missing")

    online: dict[str, float] = {}
    fallback = {
        "goodput_rps": "goodput_rps",
        "good_rate": "good_request_fraction",
        "request_throughput_rps": "request_throughput_rps",
        "completed_requests": "request_count",
        "duration_s": "benchmark_duration_s",
        "mean_ttft_ms": "mean_ttft_ms",
        "p99_ttft_ms": "p99_ttft_ms",
        "mean_itl_ms": "mean_itl_ms",
        "p99_itl_ms": "p99_itl_ms",
        "mean_e2e_ms": "mean_e2e_ms",
        "p99_e2e_ms": "p99_e2e_ms",
    }
    for name, raw_name in fallback.items():
        value = metrics.get(name, raw_metrics.get(raw_name))
        online[name] = _nonnegative_number(value, f"online metric {name}")

    summary_paths = sorted(
        (run_directory / "planner-gym" / "artifacts").rglob(
            "profile_export_aiperf.json"
        )
    )
    if len(summary_paths) != 1:
        raise CampaignCompileError(
            "expected exactly one AIPerf profile_export_aiperf.json, "
            f"found {len(summary_paths)}"
        )
    aiperf_path = summary_paths[0]
    aiperf = _read_json_object(aiperf_path)
    relative_aiperf = str(aiperf_path.relative_to(run_directory))
    record["source_files"][relative_aiperf] = _file_provenance(aiperf_path)
    successful = _aiperf_stat(aiperf, "request_count")
    if successful is None:
        raise CampaignCompileError("AIPerf request_count is missing")
    errors = _aiperf_stat(aiperf, "error_request_count")
    # AIPerf marks error_request_count ERROR_ONLY, so omission is a clean zero.
    errors = 0.0 if errors is None else errors
    error_rate_percent = _aiperf_stat(aiperf, "request_error_rate")
    attempted = successful + errors
    error_rate = (
        error_rate_percent / 100.0
        if error_rate_percent is not None
        else (errors / attempted if attempted else 0.0)
    )
    if not math.isclose(
        successful, online["completed_requests"], rel_tol=1e-9, abs_tol=1e-9
    ):
        raise CampaignCompileError(
            "Planner Gym completed_requests disagrees with AIPerf request_count"
        )
    if not 0 <= error_rate <= 1:
        raise CampaignCompileError("AIPerf request error rate is outside [0, 1]")
    online.update(error_count=errors, error_rate=error_rate)
    record["online"] = online

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise CampaignCompileError("Planner Gym provenance is missing")
    online_start = _parse_time(provenance.get("started_at"), "Planner Gym started_at")
    online_end = _parse_time(provenance.get("finished_at"), "Planner Gym finished_at")
    if online_end <= online_start:
        raise CampaignCompileError("Planner Gym interval is not positive")
    record["online_interval"] = {
        "started_at": online_start.isoformat(),
        "finished_at": online_end.isoformat(),
        "seconds": (online_end - online_start).total_seconds(),
    }

    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        raise CampaignCompileError("Planner Gym evaluation metadata is missing")
    trace = evaluation.get("trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("sha256"), str):
        raise CampaignCompileError("Planner Gym trace fingerprint is missing")
    signature = {
        "workload": evaluation.get("workload"),
        "seed": evaluation.get("seed"),
        "max_requests": evaluation.get("max_requests"),
        "arrival_speedup": evaluation.get("arrival_speedup"),
        "trace_block_size": evaluation.get("trace_block_size"),
        "trace_sha256": trace.get("sha256"),
        "trace_source_sha256": trace.get("source_sha256"),
        "sla": evaluation.get("sla"),
    }
    record["workload_signature"] = signature
    record["workload_signature_sha256"] = _canonical_sha256(signature)
    record["provenance"]["planner_gym"] = {
        "report": provenance,
        "evaluation": evaluation,
    }
    return online_start, online_end


def _completed_at(records: Sequence[dict[str, Any]], when: dt.datetime) -> int:
    completed = 0
    for item in records:
        if item["_observed_at"] > when:
            break
        completed = item["_completed"]
    return completed


def _metric_values(payload: str, metric: str) -> list[float]:
    values: list[float] = []
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([^\s]+)")
    for line in payload.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        try:
            value = float(match.group(1))
        except ValueError as error:
            raise CampaignCompileError(
                f"metric {metric} contains a non-numeric sample"
            ) from error
        if not math.isfinite(value):
            raise CampaignCompileError(f"metric {metric} contains a non-finite sample")
        values.append(value)
    return values


def _prometheus_labels(payload: str, label: str) -> dict[str, str]:
    if not payload:
        return {}
    labels: dict[str, str] = {}
    position = 0
    while position < len(payload):
        match = PROMETHEUS_LABEL_PATTERN.match(payload, position)
        if match is None:
            raise CampaignCompileError(f"{label} contains malformed Prometheus labels")
        name = match.group("name")
        if name in labels:
            raise CampaignCompileError(
                f"{label} contains duplicate Prometheus label {name!r}"
            )
        try:
            labels[name] = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError as error:
            raise CampaignCompileError(
                f"{label} contains an invalid escaped label value"
            ) from error
        position = match.end()
    return labels


def _matching_metric_values(
    payload: str,
    metric: str,
    required_labels: dict[str, str],
    label: str,
) -> list[float]:
    values: list[float] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE_PATTERN.match(line)
        if match is None:
            if line.startswith(metric):
                raise CampaignCompileError(
                    f"{label} line {line_number} is malformed Prometheus text"
                )
            continue
        if match.group("name") != metric:
            continue
        labels = _prometheus_labels(
            match.group("labels") or "", f"{label} line {line_number}"
        )
        if any(labels.get(name) != value for name, value in required_labels.items()):
            continue
        try:
            value = float(match.group("value"))
        except ValueError as error:
            raise CampaignCompileError(
                f"{label} metric {metric} contains a non-numeric sample"
            ) from error
        if not math.isfinite(value):
            raise CampaignCompileError(
                f"{label} metric {metric} contains a non-finite sample"
            )
        values.append(value)
    return values


def _load_metric_snapshots(
    evidence_root: Path,
    endpoint: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    directory = evidence_root / "metrics" / endpoint
    paths = sorted(directory.glob("*.prom"))
    if not paths:
        raise CampaignCompileError(f"no {endpoint} metric snapshots exist")
    snapshots: list[dict[str, Any]] = []
    outer_run_directory = Path(record["run_directory"])
    for path in paths:
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise CampaignCompileError(
                f"{path} is missing its MetricsSampler timestamp sidecar"
            )
        metadata = _read_json_object(metadata_path)
        observed_at = _parse_time(
            metadata.get("observed_at"), f"{metadata_path} observed_at"
        )
        snapshots.append(
            {
                "observed_at": observed_at,
                "payload": path.read_text(encoding="utf-8"),
                "path": path,
                "metadata_path": metadata_path,
            }
        )
        for source_path in (path, metadata_path):
            record["source_files"][
                str(source_path.relative_to(outer_run_directory))
            ] = _file_provenance(source_path)
    snapshots.sort(key=lambda item: item["observed_at"])
    observed_times = [item["observed_at"] for item in snapshots]
    if len(set(observed_times)) != len(observed_times):
        raise CampaignCompileError(
            f"{endpoint} metric snapshots contain duplicate observation timestamps"
        )
    return snapshots


def _single_matching_metric(
    snapshot: dict[str, Any],
    metric: str,
    labels: dict[str, str],
) -> float:
    path = snapshot["path"]
    values = _matching_metric_values(snapshot["payload"], metric, labels, str(path))
    if len(values) != 1:
        raise CampaignCompileError(
            f"{path} contains {len(values)} matching {metric} samples; expected one"
        )
    return values[0]


def _stable_frontend_phases(
    snapshots: Sequence[dict[str, Any]],
    *,
    model: str,
    online_start: dt.datetime,
    online_end: dt.datetime,
) -> dict[str, list[dict[str, Any]]]:
    labels = {
        "model": model,
        "endpoint": "chat_completions",
        "request_type": "stream",
    }
    counters: list[tuple[dt.datetime, float]] = []
    for snapshot in snapshots:
        counters.append(
            (
                snapshot["observed_at"],
                _single_matching_metric(snapshot, FRONTEND_REQUESTS_STARTED, labels),
            )
        )

    rates: list[dict[str, Any]] = []
    for (previous_at, previous), (observed_at, current) in pairwise(counters):
        elapsed = (observed_at - previous_at).total_seconds()
        if elapsed <= 0:
            raise CampaignCompileError(
                "frontend metric snapshot timestamps are not strictly increasing"
            )
        if current < previous:
            raise CampaignCompileError(
                "frontend requests_started_total decreased during the run"
            )
        if previous_at < online_start or observed_at > online_end:
            continue
        rates.append(
            {
                "interval_started_at": previous_at,
                "observed_at": observed_at,
                "rps": (current - previous) / elapsed,
            }
        )
    if not rates:
        raise CampaignCompileError(
            "no frontend requests_started_total deltas fall within the online interval"
        )

    high_candidates = [
        sample
        for sample in rates
        if HIGH_LOAD_RPS_RANGE[0] <= sample["rps"] <= HIGH_LOAD_RPS_RANGE[1]
    ]
    if len(high_candidates) < MINIMUM_STABLE_PHASE_SAMPLES:
        raise CampaignCompileError(
            "insufficient frontend samples in the observed 7-9 RPS high-load band"
        )
    spacings = [
        (current["observed_at"] - previous["observed_at"]).total_seconds()
        for previous, current in pairwise(rates)
    ]
    typical_spacing = statistics.median(spacings) if spacings else 0.0
    maximum_high_gap = max(30.0, typical_spacing * 3.0)
    high_runs: list[list[dict[str, Any]]] = []
    for sample in high_candidates:
        if (
            not high_runs
            or (
                sample["observed_at"] - high_runs[-1][-1]["observed_at"]
            ).total_seconds()
            > maximum_high_gap
        ):
            high_runs.append([sample])
        else:
            high_runs[-1].append(sample)
    qualifying_high_runs = [
        run for run in high_runs if len(run) >= MINIMUM_STABLE_PHASE_SAMPLES
    ]
    if len(qualifying_high_runs) != 1:
        raise CampaignCompileError(
            "frontend deltas do not identify exactly one sustained high-load block"
        )
    high_run = qualifying_high_runs[0]
    high_first = high_run[0]["observed_at"]
    high_last = high_run[-1]["observed_at"]
    margin = dt.timedelta(seconds=PHASE_TRANSITION_MARGIN_SECONDS)
    phase_samples = {
        "low_1": [
            sample
            for sample in rates
            if LOW_LOAD_RPS_RANGE[0] <= sample["rps"] <= LOW_LOAD_RPS_RANGE[1]
            and sample["observed_at"] <= high_first - margin
        ],
        "high": [
            sample
            for sample in high_run
            if high_first + margin <= sample["observed_at"] <= high_last - margin
        ],
        "low_2": [
            sample
            for sample in rates
            if LOW_LOAD_RPS_RANGE[0] <= sample["rps"] <= LOW_LOAD_RPS_RANGE[1]
            and sample["observed_at"] >= high_last + margin
        ],
    }
    for phase, samples in phase_samples.items():
        if len(samples) < MINIMUM_STABLE_PHASE_SAMPLES:
            raise CampaignCompileError(
                f"native control evidence has only {len(samples)} stable {phase} "
                f"frontend samples; need {MINIMUM_STABLE_PHASE_SAMPLES}"
            )
    return phase_samples


def _phase_for_time(
    observed_at: dt.datetime,
    phase_windows: dict[str, tuple[dt.datetime, dt.datetime]],
) -> str | None:
    for phase, (started_at, finished_at) in phase_windows.items():
        if started_at <= observed_at <= finished_at:
            return phase
    return None


def _planner_decisions(
    evidence_root: Path,
    native_summary: dict[str, Any],
    planner_pod: dict[str, Any],
    pool_id: str,
    phase_windows: dict[str, tuple[dt.datetime, dt.datetime]],
    record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    log_commands = native_summary.get("log_commands")
    expected_log_name = f"logs-{planner_pod['name']}"
    if not isinstance(log_commands, dict) or log_commands.get(expected_log_name) != 0:
        raise CampaignCompileError("native Planner timestamped log capture failed")
    log_path = evidence_root / "kubernetes" / "end" / f"{expected_log_name}.stdout"
    if not log_path.is_file():
        raise CampaignCompileError(f"native Planner log is missing: {log_path}")
    outer_run_directory = Path(record["run_directory"])
    record["source_files"][
        str(log_path.relative_to(outer_run_directory))
    ] = _file_provenance(log_path)

    decisions = {phase: [] for phase in phase_windows}
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = PLANNER_DECISION_PATTERN.search(line)
        if match is None or match.group("pool_id") != pool_id:
            continue
        timestamp_text = line.split(maxsplit=1)[0] if line.strip() else ""
        timestamp_text = re.sub(
            r"(\.\d{6})\d+(?=Z$|[+-]\d{2}:\d{2}$)", r"\1", timestamp_text
        )
        observed_at = _parse_time(
            timestamp_text, f"{log_path} line {line_number} timestamp"
        )
        phase = _phase_for_time(observed_at, phase_windows)
        if phase is None:
            continue
        cap = _nonnegative_number(
            float(match.group("max_admission_rps")),
            f"{log_path} line {line_number} max_admission_rps",
        )
        valid_until = _nonnegative_number(
            float(match.group("valid_until_s")),
            f"{log_path} line {line_number} valid_until_s",
        )
        decisions[phase].append(
            {
                "observed_at": observed_at,
                "max_admission_rps": cap,
                "valid_until_s": valid_until,
                "lease_unexpired": valid_until > observed_at.timestamp(),
            }
        )
    return decisions


def _async_phase_samples(
    snapshots: Sequence[dict[str, Any]],
    pool_id: str,
    phase_windows: dict[str, tuple[dt.datetime, dt.datetime]],
) -> dict[str, list[dict[str, Any]]]:
    labels = {"pool_name": pool_id}
    samples = {phase: [] for phase in phase_windows}
    for snapshot in snapshots:
        observed_at = snapshot["observed_at"]
        phase = _phase_for_time(observed_at, phase_windows)
        if phase is None:
            continue
        path = str(snapshot["path"])

        required_values: dict[str, list[float]] = {}
        for metric in (
            ASYNC_DRAIN_LIMIT_RPS,
            ASYNC_DRAIN_LIMIT_LEASE_VALID,
            ASYNC_DRAIN_LIMIT_VALID_UNTIL,
            ASYNC_BACKLOG_SOURCE_AVAILABLE,
            ASYNC_BROKER_BACKLOG,
        ):
            matched = _matching_metric_values(snapshot["payload"], metric, labels, path)
            if not matched:
                raise CampaignCompileError(
                    f"{path} has no {metric} sample for pool {pool_id!r}"
                )
            required_values[metric] = matched

        caps = required_values[ASYNC_DRAIN_LIMIT_RPS]
        lease_valid = required_values[ASYNC_DRAIN_LIMIT_LEASE_VALID]
        valid_until = required_values[ASYNC_DRAIN_LIMIT_VALID_UNTIL]
        source_available = required_values[ASYNC_BACKLOG_SOURCE_AVAILABLE]
        backlog = required_values[ASYNC_BROKER_BACKLOG]
        queue_depth = _matching_metric_values(
            snapshot["payload"], ASYNC_QUEUE_DEPTH, labels, path
        )
        for metric, metric_values in (
            (ASYNC_DRAIN_LIMIT_RPS, caps),
            (ASYNC_DRAIN_LIMIT_VALID_UNTIL, valid_until),
            (ASYNC_BROKER_BACKLOG, backlog),
            (ASYNC_QUEUE_DEPTH, queue_depth),
        ):
            if any(value < 0 for value in metric_values):
                raise CampaignCompileError(
                    f"{path} metric {metric} contains a negative sample"
                )
        samples[phase].append(
            {
                "observed_at": observed_at,
                "max_admission_rps": min(caps),
                "lease_valid": bool(lease_valid)
                and all(value == 1.0 for value in lease_valid),
                "valid_until_s": min(valid_until),
                "lease_unexpired": min(valid_until) > observed_at.timestamp(),
                "backlog_source_available": bool(source_available)
                and all(value == 1.0 for value in source_available),
                "queued_requests": sum(backlog) + sum(queue_depth),
            }
        )
    return samples


def _native_batch_drain_control_evidence(
    evidence_root: Path,
    native_summary: dict[str, Any],
    planner_pod: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    outer_configuration = record["provenance"]["outer"].get("configuration")
    if not isinstance(outer_configuration, dict):
        raise CampaignCompileError("outer configuration is missing")
    model = outer_configuration.get("model")
    pool_id = outer_configuration.get("expected_worker_pool_id")
    if not isinstance(model, str) or not model:
        raise CampaignCompileError("outer model is missing")
    if not isinstance(pool_id, str) or not pool_id:
        raise CampaignCompileError("outer expected_worker_pool_id is missing")
    online_interval = record.get("online_interval")
    if not isinstance(online_interval, dict):
        raise CampaignCompileError("online interval is missing")
    online_start = _parse_time(
        online_interval.get("started_at"), "online interval started_at"
    )
    online_end = _parse_time(
        online_interval.get("finished_at"), "online interval finished_at"
    )

    frontend_snapshots = _load_metric_snapshots(evidence_root, "frontend", record)
    frontend_phases = _stable_frontend_phases(
        frontend_snapshots,
        model=model,
        online_start=online_start,
        online_end=online_end,
    )
    phase_windows = {
        phase: (samples[0]["observed_at"], samples[-1]["observed_at"])
        for phase, samples in frontend_phases.items()
    }
    decisions = _planner_decisions(
        evidence_root,
        native_summary,
        planner_pod,
        pool_id,
        phase_windows,
        record,
    )
    async_snapshots = _load_metric_snapshots(evidence_root, "async", record)
    applied = _async_phase_samples(async_snapshots, pool_id, phase_windows)

    checks: dict[str, bool] = {}
    phase_evidence: dict[str, Any] = {}
    desired_medians: dict[str, float] = {}
    applied_medians: dict[str, float] = {}
    for phase in ("low_1", "high", "low_2"):
        frontend_rates = [sample["rps"] for sample in frontend_phases[phase]]
        phase_decisions = decisions[phase]
        phase_applied = applied[phase]
        checks[f"{phase}_minimum_frontend_samples"] = (
            len(frontend_rates) >= MINIMUM_STABLE_PHASE_SAMPLES
        )
        checks[f"{phase}_minimum_planner_decisions"] = (
            len(phase_decisions) >= MINIMUM_STABLE_PHASE_SAMPLES
        )
        checks[f"{phase}_minimum_async_samples"] = (
            len(phase_applied) >= MINIMUM_STABLE_PHASE_SAMPLES
        )
        checks[f"{phase}_planner_leases_unexpired"] = bool(phase_decisions) and all(
            sample["lease_unexpired"] for sample in phase_decisions
        )
        checks[f"{phase}_async_leases_valid_and_unexpired"] = bool(
            phase_applied
        ) and all(
            sample["lease_valid"] and sample["lease_unexpired"]
            for sample in phase_applied
        )
        checks[f"{phase}_async_backlog_available_and_positive"] = bool(
            phase_applied
        ) and all(
            sample["backlog_source_available"] and sample["queued_requests"] > 0
            for sample in phase_applied
        )
        desired_caps = [sample["max_admission_rps"] for sample in phase_decisions]
        applied_caps = [sample["max_admission_rps"] for sample in phase_applied]
        desired_summary = _metric_summary(desired_caps)
        applied_summary = _metric_summary(applied_caps)
        if desired_summary["median"] is not None:
            desired_medians[phase] = desired_summary["median"]
        if applied_summary["median"] is not None:
            applied_medians[phase] = applied_summary["median"]
        phase_evidence[phase] = {
            "window": {
                "started_at": phase_windows[phase][0].isoformat(),
                "finished_at": phase_windows[phase][1].isoformat(),
            },
            "frontend_request_rate_rps": _metric_summary(frontend_rates),
            "planner_desired_cap_rps": desired_summary,
            "async_applied_cap_rps": applied_summary,
            "planner_decision_count": len(phase_decisions),
            "planner_unexpired_lease_count": sum(
                sample["lease_unexpired"] for sample in phase_decisions
            ),
            "async_sample_count": len(phase_applied),
            "async_valid_unexpired_lease_count": sum(
                sample["lease_valid"] and sample["lease_unexpired"]
                for sample in phase_applied
            ),
            "async_active_backlog_sample_count": sum(
                sample["backlog_source_available"] and sample["queued_requests"] > 0
                for sample in phase_applied
            ),
            "async_min_queued_requests": (
                min(sample["queued_requests"] for sample in phase_applied)
                if phase_applied
                else None
            ),
            "async_min_valid_until_s": (
                min(sample["valid_until_s"] for sample in phase_applied)
                if phase_applied
                else None
            ),
        }

    if set(desired_medians) == {"low_1", "high", "low_2"}:
        checks["planner_low_caps_in_expected_range"] = all(
            LOW_DRAIN_CAP_RANGE[0] <= desired_medians[phase] <= LOW_DRAIN_CAP_RANGE[1]
            for phase in ("low_1", "low_2")
        )
        checks["planner_high_cap_in_expected_range"] = (
            desired_medians["high"] <= HIGH_DRAIN_CAP_MAXIMUM
        )
        checks["planner_cap_decreased_for_high_load"] = desired_medians["high"] < min(
            desired_medians["low_1"], desired_medians["low_2"]
        )
        checks["planner_cap_resumed_for_final_low_load"] = (
            desired_medians["low_2"] > desired_medians["high"]
        )
    else:
        checks.update(
            {
                "planner_low_caps_in_expected_range": False,
                "planner_high_cap_in_expected_range": False,
                "planner_cap_decreased_for_high_load": False,
                "planner_cap_resumed_for_final_low_load": False,
            }
        )
    if set(applied_medians) == {"low_1", "high", "low_2"}:
        checks["async_low_caps_in_expected_range"] = all(
            LOW_DRAIN_CAP_RANGE[0] <= applied_medians[phase] <= LOW_DRAIN_CAP_RANGE[1]
            for phase in ("low_1", "low_2")
        )
        checks["async_high_cap_in_expected_range"] = (
            applied_medians["high"] <= HIGH_DRAIN_CAP_MAXIMUM
        )
        checks["async_cap_decreased_for_high_load"] = applied_medians["high"] < min(
            applied_medians["low_1"], applied_medians["low_2"]
        )
        checks["async_cap_resumed_for_final_low_load"] = (
            applied_medians["low_2"] > applied_medians["high"]
        )
    else:
        checks.update(
            {
                "async_low_caps_in_expected_range": False,
                "async_high_cap_in_expected_range": False,
                "async_cap_decreased_for_high_load": False,
                "async_cap_resumed_for_final_low_load": False,
            }
        )
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "valid": not failed_checks,
        "pool_id": pool_id,
        "model": model,
        "transition_margin_seconds": PHASE_TRANSITION_MARGIN_SECONDS,
        "minimum_samples_per_stable_phase": MINIMUM_STABLE_PHASE_SAMPLES,
        "frontend_load_bands_rps": {
            "low": list(LOW_LOAD_RPS_RANGE),
            "high": list(HIGH_LOAD_RPS_RANGE),
        },
        "phases": phase_evidence,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _one_pod(summary: dict[str, Any], prefix: str, phase: str) -> dict[str, Any]:
    pod_status = summary.get("pod_status")
    if not isinstance(pod_status, list):
        raise CampaignCompileError(f"{phase} Kubernetes pod_status is missing")
    matched = [
        item
        for item in pod_status
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].startswith(prefix)
    ]
    if len(matched) != 1:
        raise CampaignCompileError(
            f"{phase} evidence expected exactly one {prefix} pod, found {len(matched)}"
        )
    pod = matched[0]
    if (
        pod.get("phase") != "Running"
        or pod.get("ready") is not True
        or pod.get("deleting") is not False
        or pod.get("restart_count") != 0
    ):
        raise CampaignCompileError(f"{phase} pod {pod.get('name')} is not clean/Ready")
    image_ids = pod.get("image_ids")
    if not isinstance(image_ids, list) or len(image_ids) != 1 or not image_ids[0]:
        raise CampaignCompileError(f"{phase} pod {pod.get('name')} imageID is missing")
    if not isinstance(pod.get("uid"), str) or not pod["uid"]:
        raise CampaignCompileError(f"{phase} pod {pod.get('name')} UID is missing")
    return pod


def _image_inventory(
    evidence_root: Path,
    phase: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    path = evidence_root / "kubernetes" / phase / "images.json"
    raw = _read_json_array(path)
    inventory: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CampaignCompileError(f"{path} item {index} is not an object")
        inventory.append(item)
    outer_run_directory = Path(record["run_directory"])
    record["source_files"][
        str(path.relative_to(outer_run_directory))
    ] = _file_provenance(path)
    return inventory


def _image_identity(
    inventory: Sequence[dict[str, Any]],
    pod: dict[str, Any],
    phase: str,
    label: str,
    image_id_attestations: dict[str, str],
    used_image_id_attestations: set[str],
) -> dict[str, Any]:
    pod_name = pod["name"]
    matched = [item for item in inventory if item.get("pod") == pod_name]
    if len(matched) != 1:
        raise CampaignCompileError(
            f"{phase} {label} pod {pod_name} expected exactly one image record, "
            f"found {len(matched)}"
        )
    item = matched[0]
    image_ref = item.get("image")
    image_id = item.get("image_id")
    if not isinstance(image_ref, str) or not image_ref:
        raise CampaignCompileError(
            f"{phase} {label} pod {pod_name} image reference is missing"
        )
    if not isinstance(image_id, str) or not image_id:
        raise CampaignCompileError(f"{phase} {label} pod {pod_name} imageID is missing")
    if (
        item.get("phase") != "Running"
        or item.get("ready") is not True
        or item.get("restart_count") != 0
    ):
        raise CampaignCompileError(
            f"{phase} {label} pod {pod_name} image record is not clean/Ready"
        )
    if pod["image_ids"] != [image_id]:
        raise CampaignCompileError(
            f"{phase} {label} pod {pod_name} image record disagrees with pod status"
        )
    image_id_attested = image_id.startswith("<redacted")
    resolved_image_id = image_id
    if image_id_attested:
        resolved_image_id = image_id_attestations.get(image_ref, "")
        if not resolved_image_id:
            raise CampaignCompileError(
                f"{phase} {label} pod {pod_name} has a sanitized imageID; "
                f"provide --image-id-attestation {image_ref}=<resolved-image-id>"
            )
        used_image_id_attestations.add(image_ref)
    return {
        "image_ref": image_ref,
        "image_id": resolved_image_id,
        "observed_image_id": image_id,
        "image_id_attested": image_id_attested,
    }


def _signed_image_identity(identity: dict[str, Any]) -> dict[str, str]:
    return {
        "image_ref": str(identity["image_ref"]),
        "image_id": str(identity["image_id"]),
    }


def _extract_runtime_evidence(
    evidence_root: Path,
    evidence_metadata: dict[str, Any],
    record: dict[str, Any],
    image_id_attestations: dict[str, str],
    used_image_id_attestations: set[str],
) -> None:
    start = evidence_metadata.get("kubernetes_start")
    end = evidence_metadata.get("kubernetes_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise CampaignCompileError("start/end Kubernetes evidence is missing")

    inventories = {
        phase: _image_inventory(evidence_root, phase, record)
        for phase in ("start", "end")
    }
    target: dict[str, dict[str, Any]] = {}
    serving_identity: dict[str, dict[str, Any]] = {}
    for prefix, label in (
        (FRONTEND_POD_PREFIX, "frontend"),
        (WORKER_POD_PREFIX, "worker"),
    ):
        first = _one_pod(start, prefix, "start")
        last = _one_pod(end, prefix, "end")
        for field in ("name", "uid", "image_ids"):
            if first.get(field) != last.get(field):
                raise CampaignCompileError(
                    f"{label} {field} changed during the online interval"
                )
        target[label] = last
        first_identity = _image_identity(
            inventories["start"],
            first,
            "start",
            label,
            image_id_attestations,
            used_image_id_attestations,
        )
        last_identity = _image_identity(
            inventories["end"],
            last,
            "end",
            label,
            image_id_attestations,
            used_image_id_attestations,
        )
        if first_identity != last_identity:
            raise CampaignCompileError(
                f"{label} image identity changed during the online interval"
            )
        serving_identity[label] = last_identity

    async_first = _one_pod(start, ASYNC_POD_PREFIX, "start")
    async_last = _one_pod(end, ASYNC_POD_PREFIX, "end")
    for field in ("name", "uid", "image_ids"):
        if async_first.get(field) != async_last.get(field):
            raise CampaignCompileError(
                f"Async {field} changed during the online interval"
            )
    async_first_identity = _image_identity(
        inventories["start"],
        async_first,
        "start",
        "Async",
        image_id_attestations,
        used_image_id_attestations,
    )
    async_identity = _image_identity(
        inventories["end"],
        async_last,
        "end",
        "Async",
        image_id_attestations,
        used_image_id_attestations,
    )
    if async_first_identity != async_identity:
        raise CampaignCompileError(
            "Async image identity changed during the online interval"
        )

    planner_start = [
        item
        for item in start.get("pod_status", [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].startswith(PLANNER_POD_PREFIX)
    ]
    planner_end = [
        item
        for item in end.get("pod_status", [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].startswith(PLANNER_POD_PREFIX)
    ]
    planner_last: dict[str, Any] | None = None
    planner_identity: dict[str, Any] | None = None
    native: dict[str, Any] | None = None
    if record["arm"] == "planner-native":
        planner_first = _one_pod(start, PLANNER_POD_PREFIX, "start")
        planner_last = _one_pod(end, PLANNER_POD_PREFIX, "end")
        for field in ("name", "uid", "image_ids"):
            if planner_first.get(field) != planner_last.get(field):
                raise CampaignCompileError(
                    f"Planner {field} changed during the online interval"
                )
        planner_first_identity = _image_identity(
            inventories["start"],
            planner_first,
            "start",
            "Planner",
            image_id_attestations,
            used_image_id_attestations,
        )
        planner_identity = _image_identity(
            inventories["end"],
            planner_last,
            "end",
            "Planner",
            image_id_attestations,
            used_image_id_attestations,
        )
        if planner_first_identity != planner_identity:
            raise CampaignCompileError(
                "Planner image identity changed during the online interval"
            )
        native = end.get("native_planner")
        if not isinstance(native, dict):
            raise CampaignCompileError("native Planner evidence is missing")
        minimum_logs = native.get("minimum_decision_logs")
        matched_logs = native.get("decision_log_match_count")
        if (
            isinstance(minimum_logs, bool)
            or not isinstance(minimum_logs, int)
            or isinstance(matched_logs, bool)
            or not isinstance(matched_logs, int)
            or matched_logs < minimum_logs
        ):
            raise CampaignCompileError(
                "native Planner decision evidence is insufficient"
            )
    elif planner_start or planner_end:
        raise CampaignCompileError(
            f"{record['arm']} evidence unexpectedly contains a Planner pod"
        )

    for phase, inventory in inventories.items():
        planner_images = [
            item
            for item in inventory
            if isinstance(item.get("pod"), str)
            and item["pod"].startswith(PLANNER_POD_PREFIX)
        ]
        if record["arm"] != "planner-native" and planner_images:
            raise CampaignCompileError(
                f"{record['arm']} {phase} image inventory unexpectedly contains "
                "a Planner pod"
            )

    metrics_summary = evidence_metadata.get("metrics_summary")
    if (
        not isinstance(metrics_summary, dict)
        or metrics_summary.get("enabled") is not True
        or metrics_summary.get("error") is not None
        or not isinstance(metrics_summary.get("samples"), int)
        or metrics_summary["samples"] < 2
    ):
        raise CampaignCompileError("continuous metrics evidence is missing or failed")

    frontend_metrics = sorted((evidence_root / "metrics" / "frontend").glob("*.prom"))
    if len(frontend_metrics) < 2:
        raise CampaignCompileError("fewer than two frontend metric snapshots exist")
    process_starts: set[float] = set()
    outer_run_directory = Path(record["run_directory"])
    for path in frontend_metrics:
        payload = path.read_text(encoding="utf-8")
        starts = _metric_values(payload, "process_start_time_seconds")
        ready = _metric_values(payload, "dynamo_frontend_model_ready")
        if len(starts) != 1:
            raise CampaignCompileError(
                f"{path} does not contain exactly one process_start_time_seconds"
            )
        if not ready or max(ready) != 1.0:
            raise CampaignCompileError(
                f"{path} does not prove dynamo_frontend_model_ready=1"
            )
        process_starts.add(starts[0])
        record["source_files"][
            str(path.relative_to(outer_run_directory))
        ] = _file_provenance(path)
    if len(process_starts) != 1:
        raise CampaignCompileError("frontend process_start_time_seconds changed in-run")

    serving_signature_identity = {
        role: _signed_image_identity(identity)
        for role, identity in serving_identity.items()
    }
    control_plane_identity = {"async": async_identity}
    if planner_identity is not None:
        control_plane_identity["planner"] = planner_identity
    control_plane_signature_identity = {
        role: _signed_image_identity(identity)
        for role, identity in control_plane_identity.items()
    }
    record["serving_signature_sha256"] = _canonical_sha256(serving_signature_identity)
    # Retain the original field as a compatibility alias for downstream readers.
    record["deployment_signature_sha256"] = record["serving_signature_sha256"]
    record["async_signature_sha256"] = _canonical_sha256(
        _signed_image_identity(async_identity)
    )
    record["planner_signature_sha256"] = (
        _canonical_sha256(_signed_image_identity(planner_identity))
        if planner_identity is not None
        else None
    )
    record["control_plane_signature_sha256"] = _canonical_sha256(
        control_plane_signature_identity
    )
    record["evidence"] = {
        "frontend_metric_snapshot_count": len(frontend_metrics),
        "frontend_process_start_time_seconds": next(iter(process_starts)),
        "target_pods": target,
        "control_plane_pods": {
            "async": async_last,
            **({"planner": planner_last} if planner_last is not None else {}),
        },
        "deployment_identity": {
            "serving": serving_identity,
            "control_plane": control_plane_identity,
        },
        "metrics_summary": metrics_summary,
    }
    if record["arm"] == "planner-native":
        assert planner_last is not None
        assert native is not None
        try:
            control = _native_batch_drain_control_evidence(
                evidence_root, native, planner_last, record
            )
        except (CampaignCompileError, OSError, ValueError) as error:
            record["evidence"]["batch_drain_control"] = {
                "valid": False,
                "error": str(error),
            }
            raise CampaignCompileError(
                f"native batch drain control evidence is invalid: {error}"
            ) from error
        record["evidence"]["batch_drain_control"] = control
        if not control["valid"]:
            raise CampaignCompileError(
                "native batch drain control evidence failed checks: "
                + ", ".join(control["failed_checks"])
            )


def _extract_batch(
    run_directory: Path,
    record: dict[str, Any],
    metadata: dict[str, Any],
    online_start: dt.datetime,
    online_end: dt.datetime,
    minimum_overlap_fraction: float,
    image_id_attestations: dict[str, str],
    used_image_id_attestations: set[str],
) -> None:
    child_root = run_directory / "batch-harness" / "results" / "raw"
    child_runs = (
        sorted(path for path in child_root.iterdir() if path.is_dir())
        if child_root.is_dir()
        else []
    )
    if len(child_runs) != 1:
        raise CampaignCompileError(
            f"expected exactly one nested Batch run, found {len(child_runs)}"
        )
    child = child_runs[0]
    child_metadata_path = child / "metadata.json"
    child_metadata = _read_json_object(child_metadata_path)
    record["source_files"][
        str(child_metadata_path.relative_to(run_directory))
    ] = _file_provenance(child_metadata_path)
    if (
        child_metadata.get("status") != "completed"
        or child_metadata.get("exit_code") != 0
    ):
        raise CampaignCompileError("nested Batch run did not complete successfully")
    expected_kind = "baseline" if record["arm"] == "stock" else "planner-native"
    if child_metadata.get("requested_run_kind") != expected_kind:
        raise CampaignCompileError("nested Batch run kind does not match the outer arm")

    workload_manifest_path = child / "workload-manifest.json"
    workload_manifest = _read_json_object(workload_manifest_path)
    record["source_files"][
        str(workload_manifest_path.relative_to(run_directory))
    ] = _file_provenance(workload_manifest_path)
    source_sha256 = workload_manifest.get("source_sha256")
    if not isinstance(source_sha256, str) or not source_sha256:
        raise CampaignCompileError("nested Batch workload source_sha256 is missing")
    selection = workload_manifest.get("selection")
    if not isinstance(selection, dict):
        raise CampaignCompileError("nested Batch workload selection is missing")
    start_index = selection.get("start_index")
    count = selection.get("count")
    order = selection.get("order")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
    ):
        raise CampaignCompileError(
            "nested Batch workload selection start_index is invalid"
        )
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise CampaignCompileError("nested Batch workload selection count is invalid")
    if not isinstance(order, str) or not order:
        raise CampaignCompileError("nested Batch workload selection order is missing")
    request_configuration = workload_manifest.get("fixed_request_configuration")
    if not isinstance(request_configuration, dict):
        raise CampaignCompileError(
            "nested Batch fixed request configuration is missing"
        )
    method = request_configuration.get("method")
    url = request_configuration.get("url")
    model = request_configuration.get("model")
    max_tokens = request_configuration.get("max_tokens")
    temperature = request_configuration.get("temperature")
    stream = request_configuration.get("stream")
    if not isinstance(method, str) or not method:
        raise CampaignCompileError("nested Batch request method is missing")
    if not isinstance(url, str) or not url:
        raise CampaignCompileError("nested Batch request URL is missing")
    if not isinstance(model, str) or not model:
        raise CampaignCompileError("nested Batch request model is missing")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise CampaignCompileError("nested Batch request max_tokens is invalid")
    normalized_temperature = _finite_number(
        temperature, "nested Batch request temperature"
    )
    if not isinstance(stream, bool):
        raise CampaignCompileError("nested Batch request stream is not boolean")
    batch_workload_signature = {
        "source_sha256": source_sha256,
        "selection": {
            "start_index": start_index,
            "count": count,
            "order": order,
        },
        "fixed_request_configuration": {
            "method": method,
            "url": url,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": normalized_temperature,
            "stream": stream,
        },
    }
    record["batch_workload_signature"] = batch_workload_signature
    record["batch_workload_signature_sha256"] = _canonical_sha256(
        batch_workload_signature
    )
    _extract_runtime_evidence(
        child,
        child_metadata,
        record,
        image_id_attestations,
        used_image_id_attestations,
    )

    validation_path = child / "result-validation.json"
    validation = _read_json_object(validation_path)
    record["source_files"][
        str(validation_path.relative_to(run_directory))
    ] = _file_provenance(validation_path)
    if validation.get("valid") is not True:
        raise CampaignCompileError("nested Batch result validation did not pass")

    progress_path = child / "progress.jsonl"
    progress = _read_jsonl(progress_path)
    record["source_files"][
        str(progress_path.relative_to(run_directory))
    ] = _file_provenance(progress_path)
    if not progress:
        raise CampaignCompileError("nested Batch progress is empty")

    normalized: list[dict[str, Any]] = []
    prior_elapsed = -1.0
    prior_completed = -1
    totals: set[int] = set()
    interval_rates: list[float] = []
    for index, sample in enumerate(progress):
        elapsed = _nonnegative_number(
            sample.get("elapsed_seconds"), f"progress[{index}].elapsed_seconds"
        )
        completed = sample.get("completed")
        failed = sample.get("failed")
        total = sample.get("total")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < 0
            or isinstance(failed, bool)
            or not isinstance(failed, int)
            or failed < 0
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
        ):
            raise CampaignCompileError(f"progress[{index}] has invalid counts")
        if elapsed < prior_elapsed or completed < prior_completed:
            raise CampaignCompileError("nested Batch progress is not monotonic")
        if completed + failed > total:
            raise CampaignCompileError(f"progress[{index}] counts exceed total")
        observed = _parse_time(
            sample.get("observed_at"), f"progress[{index}].observed_at"
        )
        if normalized and observed < normalized[-1]["_observed_at"]:
            raise CampaignCompileError(
                "nested Batch observation timestamps are not monotonic"
            )
        rate = sample.get("interval_completion_rate_rps")
        if rate is not None:
            interval_rates.append(
                _nonnegative_number(
                    rate, f"progress[{index}].interval_completion_rate_rps"
                )
            )
        normalized.append(
            {
                **sample,
                "_elapsed": elapsed,
                "_completed": completed,
                "_failed": failed,
                "_total": total,
                "_observed_at": observed,
            }
        )
        prior_elapsed = elapsed
        prior_completed = completed
        totals.add(total)
    if len(totals) != 1:
        raise CampaignCompileError("nested Batch total changed across progress")

    final = normalized[-1]
    if final.get("status") != "completed":
        raise CampaignCompileError("nested Batch terminal status is not completed")
    if final["_completed"] + final["_failed"] != final["_total"]:
        raise CampaignCompileError("nested Batch terminal counts are incomplete")
    if final["_total"] != count:
        raise CampaignCompileError(
            "nested Batch workload selection count disagrees with terminal progress"
        )
    if (
        validation.get("completed") != final["_completed"]
        or validation.get("failed") != final["_failed"]
    ):
        raise CampaignCompileError(
            "nested Batch validation disagrees with terminal progress"
        )

    synchronization = metadata.get("synchronization")
    if not isinstance(synchronization, dict):
        raise CampaignCompileError("outer synchronization metadata is missing")
    batch_created = _parse_time(
        synchronization.get("batch_created_observed_at"),
        "batch_created_observed_at",
    )
    if synchronization.get("batch_active_status") != "in_progress":
        raise CampaignCompileError(
            "batch_active_status does not prove an in_progress Batch job"
        )
    batch_in_progress = _parse_time(
        synchronization.get("batch_active_observed_at"),
        "batch_active_observed_at",
    )
    if batch_in_progress < batch_created:
        raise CampaignCompileError(
            "Batch active observation precedes the creation observation"
        )
    if synchronization.get("batch_dispatch_active_status") != "in_progress":
        raise CampaignCompileError(
            "batch_dispatch_active_status does not prove in_progress dispatch"
        )
    dispatch_start = _parse_time(
        synchronization.get("batch_dispatch_active_observed_at"),
        "batch_dispatch_active_observed_at",
    )
    if dispatch_start < batch_in_progress:
        raise CampaignCompileError(
            "Batch dispatch-active observation precedes the in_progress observation"
        )
    if dispatch_start > online_start:
        raise CampaignCompileError(
            "Batch dispatch-active observation follows the online interval start"
        )
    dispatch_counts: dict[str, int] = {}
    for field in ("completed", "failed", "total"):
        value = synchronization.get(f"batch_dispatch_{field}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CampaignCompileError(
                f"batch_dispatch_{field} is not a non-negative integer"
            )
        dispatch_counts[field] = value
    if dispatch_counts["total"] <= 0:
        raise CampaignCompileError("batch_dispatch_total is not positive")
    if (
        not 0
        < dispatch_counts["completed"] + dispatch_counts["failed"]
        <= (dispatch_counts["total"])
    ):
        raise CampaignCompileError(
            "Batch dispatch-active counts do not prove positive bounded progress"
        )
    if not any(
        sample.get("status") == "in_progress"
        and sample["_completed"] == dispatch_counts["completed"]
        and sample["_failed"] == dispatch_counts["failed"]
        and sample["_total"] == dispatch_counts["total"]
        for sample in normalized
    ):
        raise CampaignCompileError(
            "nested Batch progress does not contain the dispatch-active observation"
        )
    batch_end = final["_observed_at"]
    if batch_end <= dispatch_start:
        raise CampaignCompileError("Batch dispatch-active interval is not positive")
    overlap_start = max(dispatch_start, online_start)
    overlap_end = min(batch_end, online_end)
    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
    if overlap_seconds <= 0:
        raise CampaignCompileError("Batch and online intervals do not overlap")
    online_seconds = (online_end - online_start).total_seconds()
    overlap_fraction = overlap_seconds / online_seconds
    if overlap_fraction < minimum_overlap_fraction:
        raise CampaignCompileError(
            "Batch/online overlap fraction "
            f"{overlap_fraction:.6f} is below required "
            f"{minimum_overlap_fraction:.6f}"
        )
    completed_start = _completed_at(normalized, overlap_start)
    completed_end = _completed_at(normalized, overlap_end)
    completed_during = max(0, completed_end - completed_start)
    duration = final["_elapsed"]
    batch = {
        "progress_sample_count": len(normalized),
        "terminal_status": final.get("status"),
        "total": final["_total"],
        "completed": final["_completed"],
        "failed": final["_failed"],
        "duration_seconds": duration,
        "average_completion_rate_rps": (
            final["_completed"] / duration if duration > 0 else None
        ),
        "peak_interval_completion_rate_rps": (
            max(interval_rates) if interval_rates else None
        ),
        "created_observed_at": batch_created.isoformat(),
        "in_progress_observed_at": batch_in_progress.isoformat(),
        "active_started_at": dispatch_start.isoformat(),
        "active_status": "in_progress",
        "dispatch_active_counts": dispatch_counts,
        "terminal_observed_at": batch_end.isoformat(),
        "overlap_seconds": overlap_seconds,
        "overlap_fraction_of_online": overlap_fraction,
        "completed_at_overlap_start": completed_start,
        "completed_at_overlap_end": completed_end,
        "completions_during_online": completed_during,
        "completion_rate_during_online_rps": completed_during / overlap_seconds,
    }
    for name in BATCH_METRICS:
        if batch.get(name) is None:
            raise CampaignCompileError(f"batch metric {name} is unavailable")
    record["batch"] = batch
    record["provenance"]["batch_harness"] = {
        "run_id": child_metadata.get("run_id"),
        "source": child_metadata.get("source"),
        "inputs": child_metadata.get("inputs"),
        "control_plane": child_metadata.get("control_plane"),
        "result_validation": validation,
    }


def _compile_one_run(
    run_directory: Path,
    minimum_batch_overlap_fraction: float,
    image_id_attestations: dict[str, str],
    used_image_id_attestations: set[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": run_directory.name,
        "run_directory": str(run_directory),
        "arm": None,
        "started_at": None,
        "ordinal_within_arm": None,
        "valid": True,
        "invalid_reasons": [],
        "online": None,
        "online_interval": None,
        "batch": None,
        "workload_signature": None,
        "workload_signature_sha256": None,
        "batch_workload_signature": None,
        "batch_workload_signature_sha256": None,
        "serving_signature_sha256": None,
        "deployment_signature_sha256": None,
        "async_signature_sha256": None,
        "planner_signature_sha256": None,
        "control_plane_signature_sha256": None,
        "evidence": None,
        "provenance": {},
        "source_files": {},
    }
    metadata_path = run_directory / "metadata.json"
    try:
        if not run_directory.is_dir():
            raise CampaignCompileError("input is not a directory")
        metadata = _read_json_object(metadata_path)
        record["source_files"]["metadata.json"] = _file_provenance(metadata_path)
        record["run_id"] = metadata.get("run_id", run_directory.name)
        record["arm"] = metadata.get("arm")
        record["started_at"] = metadata.get("started")
        if metadata.get("run_id") != run_directory.name:
            _add_invalid(record, "metadata run_id does not match directory name")
        if metadata.get("kind") != "planner-gym-batch-impact":
            _add_invalid(record, "metadata kind is not planner-gym-batch-impact")
        if record["arm"] not in ARMS:
            _add_invalid(record, f"unsupported arm {record['arm']!r}")
        if metadata.get("status") != "completed" or metadata.get("exit_code") != 0:
            _add_invalid(record, "outer run did not complete successfully")
        _parse_time(metadata.get("started"), "outer started")
        record["provenance"]["outer"] = {
            "source": metadata.get("source"),
            "inputs": metadata.get("inputs"),
            "configuration": metadata.get("configuration"),
            "synchronization": metadata.get("synchronization"),
        }
        online_start, online_end = _extract_online(run_directory, record)
        if record["arm"] in {"stock", "planner-native"}:
            _extract_batch(
                run_directory,
                record,
                metadata,
                online_start,
                online_end,
                minimum_batch_overlap_fraction,
                image_id_attestations,
                used_image_id_attestations,
            )
        else:
            child_root = run_directory / "batch-harness" / "results" / "raw"
            if child_root.is_dir() and any(child_root.iterdir()):
                _add_invalid(
                    record, "online-only arm unexpectedly contains a nested Batch run"
                )
            _extract_runtime_evidence(
                run_directory / "online-evidence",
                metadata,
                record,
                image_id_attestations,
                used_image_id_attestations,
            )
        _validate_outer_checksum_manifest(run_directory, record)
    except (CampaignCompileError, OSError) as error:
        _add_invalid(record, str(error))
    return record


def _started_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    started = record.get("started_at")
    return (started if isinstance(started, str) else "", str(record["run_id"]))


def _summarize_arm(records: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    all_records = [record for record in records if record.get("arm") == arm]
    valid = [record for record in all_records if record["valid"]]
    online = {
        name: _metric_summary(record["online"][name] for record in valid)
        for name in ONLINE_METRICS
    }
    batch = (
        {
            name: _metric_summary(record["batch"][name] for record in valid)
            for name in BATCH_METRICS
        }
        if arm != "online-only"
        else None
    )
    return {
        "run_count": len(all_records),
        "valid_run_count": len(valid),
        "invalid_run_count": len(all_records) - len(valid),
        "valid_run_ids": [record["run_id"] for record in valid],
        "invalid_run_ids": [
            record["run_id"] for record in all_records if not record["valid"]
        ],
        "online": online,
        "batch": batch,
    }


def _metric_comparison(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    section: str,
) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    for reference, treatment in pairs:
        reference_value = float(reference[section][metric])
        treatment_value = float(treatment[section][metric])
        delta = treatment_value - reference_value
        relative = (
            delta / abs(reference_value) * 100.0 if reference_value != 0 else None
        )
        improvement = -delta if metric in LOWER_IS_BETTER else delta
        values.append(
            {
                "reference": reference_value,
                "treatment": treatment_value,
                "treatment_minus_reference": delta,
                "relative_change_percent": relative,
                "improvement": improvement,
            }
        )
    deltas = [value["treatment_minus_reference"] for value in values]
    relative_changes = [
        value["relative_change_percent"]
        for value in values
        if value["relative_change_percent"] is not None
    ]
    improvements = [value["improvement"] for value in values]
    signs = {1 if delta > 0 else -1 for delta in deltas if delta != 0}
    return {
        "pair_count": len(values),
        "reference": _metric_summary(value["reference"] for value in values),
        "treatment": _metric_summary(value["treatment"] for value in values),
        "treatment_minus_reference": _metric_summary(deltas),
        "relative_change_percent": _metric_summary(relative_changes),
        "improvement": _metric_summary(improvements),
        "direction_changed_across_repetitions": len(signs) > 1,
    }


def _compile_comparison(
    records: Sequence[dict[str, Any]], reference_arm: str, treatment_arm: str
) -> dict[str, Any]:
    reference = {
        record["ordinal_within_arm"]: record
        for record in records
        if record.get("arm") == reference_arm
    }
    treatment = {
        record["ordinal_within_arm"]: record
        for record in records
        if record.get("arm") == treatment_arm
    }
    ordinals = sorted(set(reference) | set(treatment))
    valid_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pair_rows: list[dict[str, Any]] = []
    for ordinal in ordinals:
        reference_record = reference.get(ordinal)
        treatment_record = treatment.get(ordinal)
        reasons: list[str] = []
        if reference_record is None:
            reasons.append(f"missing {reference_arm} counterpart")
        elif not reference_record["valid"]:
            reasons.append(f"{reference_arm} counterpart is invalid")
        if treatment_record is None:
            reasons.append(f"missing {treatment_arm} counterpart")
        elif not treatment_record["valid"]:
            reasons.append(f"{treatment_arm} counterpart is invalid")
        if (
            not reasons
            and reference_record["workload_signature_sha256"]
            != treatment_record["workload_signature_sha256"]
        ):
            reasons.append("workload signatures differ")
        if (
            reference_record is not None
            and treatment_record is not None
            and reference_arm != "online-only"
            and treatment_arm != "online-only"
            and reference_record["batch_workload_signature_sha256"] is not None
            and treatment_record["batch_workload_signature_sha256"] is not None
            and reference_record["batch_workload_signature_sha256"]
            != treatment_record["batch_workload_signature_sha256"]
        ):
            reasons.append("Batch workload signatures differ")
        valid = not reasons
        if valid:
            valid_pairs.append((reference_record, treatment_record))
        pair_rows.append(
            {
                "ordinal": ordinal,
                "reference_run_id": (
                    reference_record["run_id"] if reference_record else None
                ),
                "treatment_run_id": (
                    treatment_record["run_id"] if treatment_record else None
                ),
                "valid": valid,
                "invalid_reasons": reasons,
            }
        )

    online = {
        metric: _metric_comparison(valid_pairs, metric, "online")
        for metric in ONLINE_METRICS
    }
    batch = None
    if reference_arm != "online-only" and treatment_arm != "online-only":
        batch = {
            metric: _metric_comparison(valid_pairs, metric, "batch")
            for metric in BATCH_METRICS
        }
    return {
        "reference_arm": reference_arm,
        "treatment_arm": treatment_arm,
        "pairing_method": "chronological ordinal within each arm",
        "candidate_pair_count": len(pair_rows),
        "valid_pair_count": len(valid_pairs),
        "invalid_pair_count": len(pair_rows) - len(valid_pairs),
        "pairs": pair_rows,
        "online": online,
        "batch": batch,
    }


def _stopping_assessment(
    arms: dict[str, Any], comparisons: dict[str, Any], minimum: int, cv_limit: float
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    instability_triggered = False
    valid_counts = [arms[arm]["valid_run_count"] for arm in ARMS]
    minimum_valid_arm_count = min(valid_counts)
    for arm in ARMS:
        valid_count = arms[arm]["valid_run_count"]
        if valid_count < minimum:
            reasons.append(
                {
                    "code": "insufficient-valid-repetitions",
                    "arm": arm,
                    "valid": valid_count,
                    "required": minimum,
                }
            )
        for metric in DECISION_ONLINE_METRICS:
            cv = arms[arm]["online"][metric]["cv_percent"]
            if cv is not None and cv > cv_limit:
                instability_triggered = True
                reasons.append(
                    {
                        "code": "cv-above-threshold",
                        "arm": arm,
                        "section": "online",
                        "metric": metric,
                        "cv_percent": cv,
                        "threshold_percent": cv_limit,
                    }
                )
        if arm != "online-only":
            for metric in DECISION_BATCH_METRICS:
                cv = arms[arm]["batch"][metric]["cv_percent"]
                if cv is not None and cv > cv_limit:
                    instability_triggered = True
                    reasons.append(
                        {
                            "code": "cv-above-threshold",
                            "arm": arm,
                            "section": "batch",
                            "metric": metric,
                            "cv_percent": cv,
                            "threshold_percent": cv_limit,
                        }
                    )

    for name, comparison in comparisons.items():
        for metric in DECISION_ONLINE_METRICS:
            if comparison["online"][metric]["direction_changed_across_repetitions"]:
                instability_triggered = True
                reasons.append(
                    {
                        "code": "paired-effect-direction-changed",
                        "comparison": name,
                        "section": "online",
                        "metric": metric,
                    }
                )
        if comparison["batch"] is not None:
            for metric in DECISION_BATCH_METRICS:
                if comparison["batch"][metric]["direction_changed_across_repetitions"]:
                    instability_triggered = True
                    reasons.append(
                        {
                            "code": "paired-effect-direction-changed",
                            "comparison": name,
                            "section": "batch",
                            "metric": metric,
                        }
                    )
    target_repetitions = max(minimum, 5) if instability_triggered else minimum
    additional = max(0, target_repetitions - minimum_valid_arm_count)
    return {
        "minimum_valid_repetitions_per_arm": minimum,
        "minimum_valid_arm_count": minimum_valid_arm_count,
        "instability_triggered": instability_triggered,
        "target_valid_repetitions_per_arm": target_repetitions,
        "cv_threshold_percent": cv_limit,
        "extra_repetitions_recommended": additional > 0,
        "minimum_additional_balanced_repetitions": additional,
        "reasons": reasons,
    }


def _enforce_signature_consistency(
    records: Sequence[dict[str, Any]],
    field: str,
    mismatch_reason: str,
) -> str | None:
    signatures = [
        record[field]
        for record in records
        if record["valid"] and record[field] is not None
    ]
    if not signatures:
        return None
    counts = Counter(signatures)
    canonical = min(counts, key=lambda value: (-counts[value], value))
    for record in records:
        if record["valid"] and record[field] != canonical:
            _add_invalid(record, mismatch_reason)
    return canonical


def _identity_for_signature(
    records: Sequence[dict[str, Any]],
    field: str,
    signature: str | None,
    identity_group: str,
) -> dict[str, Any] | None:
    if signature is None:
        return None
    for record in records:
        evidence = record.get("evidence")
        if record.get(field) != signature or not isinstance(evidence, dict):
            continue
        deployment_identity = evidence.get("deployment_identity")
        if not isinstance(deployment_identity, dict):
            continue
        identity = deployment_identity.get(identity_group)
        if isinstance(identity, dict):
            return identity
    return None


def compile_campaign(
    run_directories: Sequence[Path],
    *,
    minimum_repetitions: int = 3,
    cv_threshold_percent: float = 5.0,
    minimum_batch_overlap_fraction: float = DEFAULT_MINIMUM_BATCH_OVERLAP_FRACTION,
    image_id_attestations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compile run directories into a machine-readable campaign report."""
    if not run_directories:
        raise CampaignCompileError("at least one run directory is required")
    if minimum_repetitions <= 0:
        raise CampaignCompileError("minimum repetitions must be positive")
    if not math.isfinite(cv_threshold_percent) or cv_threshold_percent < 0:
        raise CampaignCompileError("CV threshold must be finite and non-negative")
    if (
        not math.isfinite(minimum_batch_overlap_fraction)
        or not 0 <= minimum_batch_overlap_fraction <= 1
    ):
        raise CampaignCompileError(
            "minimum Batch overlap fraction must be finite and in [0, 1]"
        )
    attestations = dict(image_id_attestations or {})
    for image_ref, image_id in attestations.items():
        if not isinstance(image_ref, str) or not image_ref:
            raise CampaignCompileError("image ID attestation has an empty reference")
        if (
            not isinstance(image_id, str)
            or not image_id
            or image_id.startswith("<redacted")
        ):
            raise CampaignCompileError(
                f"image ID attestation for {image_ref!r} is empty or redacted"
            )
    resolved = [path.expanduser().resolve() for path in run_directories]
    if len(set(resolved)) != len(resolved):
        raise CampaignCompileError("run directories contain duplicates")
    used_attestations: set[str] = set()
    records = [
        _compile_one_run(
            path,
            minimum_batch_overlap_fraction,
            attestations,
            used_attestations,
        )
        for path in resolved
    ]
    unused_attestations = sorted(set(attestations) - used_attestations)
    if unused_attestations:
        raise CampaignCompileError(
            "image ID attestation reference was not used by any sanitized deployed "
            "image: " + ", ".join(unused_attestations)
        )
    records.sort(key=_started_sort_key)
    for arm in ARMS:
        arm_records = [record for record in records if record.get("arm") == arm]
        for ordinal, record in enumerate(arm_records, start=1):
            record["ordinal_within_arm"] = ordinal

    valid_signatures = [
        record["workload_signature_sha256"]
        for record in records
        if record["valid"] and record["workload_signature_sha256"] is not None
    ]
    canonical_signature = None
    if valid_signatures:
        counts = Counter(valid_signatures)
        canonical_signature = min(counts, key=lambda value: (-counts[value], value))
        for record in records:
            if (
                record["valid"]
                and record["workload_signature_sha256"] != canonical_signature
            ):
                _add_invalid(
                    record,
                    "workload signature differs from the campaign majority",
                )

    batch_records = [
        record for record in records if record.get("arm") in {"stock", "planner-native"}
    ]
    canonical_batch_workload_signature = _enforce_signature_consistency(
        batch_records,
        "batch_workload_signature_sha256",
        "Batch workload signature differs from the campaign majority",
    )

    canonical_serving_signature = _enforce_signature_consistency(
        records,
        "serving_signature_sha256",
        "frontend/worker image identity differs from the campaign majority",
    )
    canonical_async_signatures: dict[str, str | None] = {}
    canonical_planner_signatures: dict[str, str | None] = {}
    canonical_control_plane_signatures: dict[str, str | None] = {}
    canonical_control_plane_identities: dict[str, dict[str, Any] | None] = {}
    for arm in ARMS:
        arm_records = [record for record in records if record.get("arm") == arm]
        canonical_async_signatures[arm] = _enforce_signature_consistency(
            arm_records,
            "async_signature_sha256",
            f"Async image identity differs from the {arm} arm majority",
        )
        canonical_control_plane_signatures[arm] = _enforce_signature_consistency(
            arm_records,
            "control_plane_signature_sha256",
            f"control-plane image identity differs from the {arm} arm majority",
        )
        canonical_control_plane_identities[arm] = _identity_for_signature(
            arm_records,
            "control_plane_signature_sha256",
            canonical_control_plane_signatures[arm],
            "control_plane",
        )
    native_records = [
        record for record in records if record.get("arm") == "planner-native"
    ]
    canonical_planner_signatures["planner-native"] = _enforce_signature_consistency(
        native_records,
        "planner_signature_sha256",
        "Planner image identity differs from the planner-native arm majority",
    )
    canonical_serving_identity = _identity_for_signature(
        records,
        "serving_signature_sha256",
        canonical_serving_signature,
        "serving",
    )

    arms = {arm: _summarize_arm(records, arm) for arm in ARMS}
    comparisons = {
        "stock_vs_online-only": _compile_comparison(records, "online-only", "stock"),
        "planner-native_vs_online-only": _compile_comparison(
            records, "online-only", "planner-native"
        ),
        "planner-native_vs_stock": _compile_comparison(
            records, "stock", "planner-native"
        ),
    }
    stopping = _stopping_assessment(
        arms, comparisons, minimum_repetitions, cv_threshold_percent
    )
    issues = [
        {
            "run_id": record["run_id"],
            "arm": record["arm"],
            "reasons": record["invalid_reasons"],
        }
        for record in records
        if not record["valid"]
    ]
    for arm in ARMS:
        if arms[arm]["run_count"] == 0:
            issues.append({"run_id": None, "arm": arm, "reasons": ["arm is missing"]})
    for name, comparison in comparisons.items():
        invalid_pairs = [pair for pair in comparison["pairs"] if not pair["valid"]]
        if invalid_pairs:
            issues.append(
                {
                    "run_id": None,
                    "arm": None,
                    "comparison": name,
                    "reasons": [
                        f"ordinal {pair['ordinal']}: "
                        + "; ".join(pair["invalid_reasons"])
                        for pair in invalid_pairs
                    ],
                }
            )
    compiler_path = Path(__file__).resolve()
    return {
        "schema_version": "1.0",
        "analysis_kind": "planner-gym-batch-impact-campaign",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "valid" if not issues else "invalid-inputs-present",
        "campaign": {
            "input_run_count": len(records),
            "valid_run_count": sum(record["valid"] for record in records),
            "invalid_run_count": sum(not record["valid"] for record in records),
            "canonical_workload_signature_sha256": canonical_signature,
            "canonical_batch_workload_signature_sha256": (
                canonical_batch_workload_signature
            ),
            "canonical_serving_signature_sha256": canonical_serving_signature,
            # Retain the original field as a compatibility alias.
            "canonical_deployment_signature_sha256": canonical_serving_signature,
            "canonical_async_signature_sha256_by_arm": canonical_async_signatures,
            "canonical_planner_signature_sha256_by_arm": (canonical_planner_signatures),
            "canonical_control_plane_signature_sha256_by_arm": (
                canonical_control_plane_signatures
            ),
            "canonical_image_identities": {
                "serving": canonical_serving_identity,
                "control_plane_by_arm": canonical_control_plane_identities,
            },
            "pairing_method": "chronological ordinal within each arm",
            "minimum_batch_overlap_fraction": minimum_batch_overlap_fraction,
            "pairing_caveat": (
                "Supply every attempted arm run, including failed runs, so ordinal "
                "pairing retains repetition alignment."
            ),
        },
        "arms": arms,
        "comparisons": comparisons,
        "stopping_assessment": stopping,
        "data_quality": {"issue_count": len(issues), "issues": issues},
        "runs": records,
        "provenance": {
            "compiler": {
                "path": str(compiler_path),
                "sha256": _sha256_file(compiler_path),
            },
            "image_id_attestations": [
                {
                    "image_ref": image_ref,
                    "image_id": attestations[image_ref],
                    "used_for_sanitized_observation": True,
                }
                for image_ref in sorted(attestations)
            ],
            "input_directories": [str(path) for path in resolved],
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise human index for the lossless JSON report."""
    arms = report["arms"]
    stopping = report["stopping_assessment"]
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "-->",
        "",
        "# Planner Gym Batch-impact Campaign",
        "",
        f"Status: `{report['status']}`. The lossless analysis is in `summary.json`.",
        "",
        "## Per-arm results",
        "",
        "| Arm | Valid / attempted | Goodput RPS (mean) | Goodput CV | p99 TTFT ms (mean) | Error rate (mean) | Batch RPS (mean) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        summary = arms[arm]
        goodput = summary["online"]["goodput_rps"]
        ttft = summary["online"]["p99_ttft_ms"]
        error_rate = summary["online"]["error_rate"]
        batch_rps = (
            summary["batch"]["average_completion_rate_rps"]["mean"]
            if summary["batch"] is not None
            else None
        )
        lines.append(
            f"| {arm} | {summary['valid_run_count']} / {summary['run_count']} "
            f"| {_fmt(goodput['mean'])} | {_fmt(goodput['cv_percent'])}% "
            f"| {_fmt(ttft['mean'])} | {_fmt(error_rate['mean'])} "
            f"| {_fmt(batch_rps)} |"
        )

    identities = report["campaign"]["canonical_image_identities"]
    serving = identities["serving"]
    control_plane = identities["control_plane_by_arm"]
    lines.extend(
        [
            "",
            "## Runtime image identities",
            "",
            (
                "Every identity below includes the configured image reference and "
                "resolved imageID. The lossless JSON marks IDs observed directly "
                "versus IDs restored through an explicit attestation after artifact "
                "sanitization."
            ),
            "",
            "| Scope | Role | Image reference | Resolved imageID |",
            "| --- | --- | --- | --- |",
        ]
    )
    if isinstance(serving, dict):
        for role in ("frontend", "worker"):
            identity = serving.get(role)
            if isinstance(identity, dict):
                lines.append(
                    f"| all arms | {role} | `{identity.get('image_ref')}` "
                    f"| `{identity.get('image_id')}` |"
                )
    if isinstance(control_plane, dict):
        for arm in ARMS:
            identities_for_arm = control_plane.get(arm)
            if not isinstance(identities_for_arm, dict):
                continue
            for role, identity in sorted(identities_for_arm.items()):
                if isinstance(identity, dict):
                    lines.append(
                        f"| {arm} | {role} | `{identity.get('image_ref')}` "
                        f"| `{identity.get('image_id')}` |"
                    )
    lines.extend(
        [
            "",
            (
                "- Shared serving signature: "
                f"`{report['campaign']['canonical_serving_signature_sha256']}`"
            ),
            (
                "- Async signatures by arm: "
                f"`{json.dumps(report['campaign']['canonical_async_signature_sha256_by_arm'], sort_keys=True)}`"
            ),
            (
                "- Planner signatures by arm: "
                f"`{json.dumps(report['campaign']['canonical_planner_signature_sha256_by_arm'], sort_keys=True)}`"
            ),
            (
                "- Control-plane signatures by arm: "
                f"`{json.dumps(report['campaign']['canonical_control_plane_signature_sha256_by_arm'], sort_keys=True)}`"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Paired online effects",
            "",
            "Deltas are treatment minus reference. Negative latency and error deltas are improvements.",
            "",
            "| Comparison | Valid pairs | Goodput delta | p99 TTFT delta ms | Error-rate delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in report["comparisons"].items():
        online = comparison["online"]
        lines.append(
            f"| {name} | {comparison['valid_pair_count']} "
            f"| {_fmt(online['goodput_rps']['treatment_minus_reference']['mean'])} "
            f"| {_fmt(online['p99_ttft_ms']['treatment_minus_reference']['mean'])} "
            f"| {_fmt(online['error_rate']['treatment_minus_reference']['mean'])} |"
        )

    lines.extend(["", "## Data quality and stopping", ""])
    lines.append(f"- Invalid inputs: {report['data_quality']['issue_count']}")
    lines.append(
        "- Extra repetitions recommended: "
        f"{str(stopping['extra_repetitions_recommended']).lower()}"
    )
    lines.append(
        "- Minimum additional balanced repetitions: "
        f"{stopping['minimum_additional_balanced_repetitions']}"
    )
    lines.append(
        "- Target valid repetitions per arm: "
        f"{stopping['target_valid_repetitions_per_arm']}"
    )
    lines.append(f"- Stopping-rule findings: {len(stopping['reasons'])}")
    if report["data_quality"]["issues"]:
        lines.extend(["", "### Invalid inputs", ""])
        for issue in report["data_quality"]["issues"]:
            if issue.get("comparison"):
                subject = f"comparison `{issue['comparison']}`"
            else:
                subject = f"`{issue['run_id'] or '<missing>'}` ({issue['arm']})"
            lines.append(f"- {subject}: " + "; ".join(issue["reasons"]))
    lines.extend(
        [
            "",
            "Pairing uses chronological ordinal within each arm. Include every attempted run,",
            "including failed runs, to preserve repetition alignment.",
            "",
        ]
    )
    return "\n".join(lines)


def write_campaign(report: dict[str, Any], output_directory: Path) -> Path:
    """Publish JSON and Markdown together using a same-filesystem rename."""
    destination = output_directory.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        (stage / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "README.md").write_text(render_markdown(report), encoding="utf-8")
        os.rename(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return destination


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def _parse_image_id_attestations(values: Sequence[str]) -> dict[str, str]:
    attestations: dict[str, str] = {}
    for value in values:
        image_ref, separator, image_id = value.partition("=")
        if not separator or not image_ref or not image_id:
            raise CampaignCompileError(
                "image ID attestation must use IMAGE_REF=RESOLVED_IMAGE_ID"
            )
        if image_ref in attestations:
            qualifier = (
                "duplicate" if attestations[image_ref] == image_id else "conflicting"
            )
            raise CampaignCompileError(
                f"{qualifier} image ID attestation for {image_ref!r}"
            )
        attestations[image_ref] = image_id
    return attestations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        action="append",
        required=True,
        help="top-level raw run directory; repeat for every attempted arm run",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--minimum-repetitions", type=_positive_int, default=3)
    parser.add_argument("--cv-threshold-percent", type=_nonnegative_float, default=5.0)
    parser.add_argument(
        "--minimum-batch-overlap-fraction",
        type=_fraction,
        default=DEFAULT_MINIMUM_BATCH_OVERLAP_FRACTION,
        help="minimum fraction of the online window with an active Batch job",
    )
    parser.add_argument(
        "--image-id-attestation",
        action="append",
        default=[],
        metavar="IMAGE_REF=RESOLVED_IMAGE_ID",
        help=(
            "explicit resolved imageID for a selected image whose raw artifact "
            "contains a sanitizer placeholder; repeat for each affected image ref"
        ),
    )
    parser.add_argument(
        "--allow-invalid-or-incomplete",
        action="store_true",
        help=(
            "return success even when inputs are invalid or the stopping rule "
            "requires more repetitions"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        image_id_attestations = _parse_image_id_attestations(args.image_id_attestation)
        report = compile_campaign(
            args.run_directory,
            minimum_repetitions=args.minimum_repetitions,
            cv_threshold_percent=args.cv_threshold_percent,
            minimum_batch_overlap_fraction=args.minimum_batch_overlap_fraction,
            image_id_attestations=image_id_attestations,
        )
        output = write_campaign(report, args.output_directory)
    except (CampaignCompileError, FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"compiled campaign: {output}")
    incomplete = report["stopping_assessment"]["extra_repetitions_recommended"]
    invalid = report["status"] != "valid"
    if (invalid or incomplete) and not args.allow_invalid_or_incomplete:
        conditions = []
        if invalid:
            conditions.append("invalid inputs or pairs")
        if incomplete:
            conditions.append("additional repetitions required")
        print(
            "campaign is not decision-ready: "
            + "; ".join(conditions)
            + "; pass --allow-invalid-or-incomplete to accept this result",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
