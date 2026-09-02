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

"""Compile one immutable experiment run into tabular and summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from baseline_harness import (
    EXPERIMENT_ROOT,
    RUN_ID_RE,
    HarnessError,
    isoformat_utc,
    percentile,
    sha256_file,
    write_json,
    write_text,
)

PROGRESS_FIELDS = (
    "observed_at",
    "elapsed_seconds",
    "status",
    "total",
    "reported_total",
    "total_source",
    "completed",
    "failed",
    "remaining",
    "delta_completed",
    "interval_seconds",
    "interval_completion_rate_rps",
)
ONLINE_FIELDS = (
    "request_index",
    "scheduled_offset_seconds",
    "started_at",
    "ended_at",
    "queue_delay_ms",
    "http_status",
    "ok",
    "ttft_ms",
    "latency_ms",
    "stream_protocol_seen",
    "prompt_tokens",
    "completion_tokens",
    "error_type",
    "error",
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path} does not contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while preserving line-number errors."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(
                    f"{path} line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise HarnessError(f"{path} line {line_number} is not an object")
            records.append(value)
    return records


def write_csv(
    path: Path, fields: Sequence[str], records: Iterable[dict[str, Any]]
) -> None:
    """Write a stable CSV projection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_progress(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Derive batch timing and validate monotonic progress."""
    issues = []
    if not records:
        return {
            "sample_count": 0,
            "duration_seconds": None,
            "terminal_status": None,
            "total": None,
            "completed": None,
            "failed": None,
            "average_completion_rate_rps": None,
            "peak_interval_completion_rate_rps": None,
        }, issues

    previous_elapsed = -1.0
    previous_completed = -1
    totals: set[int] = set()
    interval_rates: list[float] = []
    for index, record in enumerate(records):
        elapsed = record.get("elapsed_seconds")
        completed = record.get("completed")
        total = record.get("total")
        rate = record.get("interval_completion_rate_rps")
        if not isinstance(elapsed, (int, float)):
            issues.append(f"progress sample {index} has invalid elapsed_seconds")
            continue
        if elapsed < previous_elapsed:
            issues.append(f"progress sample {index} moves backward in time")
        previous_elapsed = float(elapsed)
        if isinstance(completed, int):
            if completed < previous_completed:
                issues.append(f"progress sample {index} decreases completed count")
            previous_completed = completed
        else:
            issues.append(f"progress sample {index} has invalid completed count")
        if isinstance(total, int) and total > 0:
            totals.add(total)
        if isinstance(rate, (int, float)) and rate >= 0:
            interval_rates.append(float(rate))
    if len(totals) > 1:
        issues.append(f"progress total changed across samples: {sorted(totals)}")

    final = records[-1]
    duration = final.get("elapsed_seconds")
    completed = final.get("completed")
    average_rate = None
    if (
        isinstance(duration, (int, float))
        and duration > 0
        and isinstance(completed, int)
    ):
        average_rate = completed / duration
    return {
        "sample_count": len(records),
        "duration_seconds": duration,
        "terminal_status": final.get("status"),
        "total": final.get("total"),
        "completed": completed,
        "failed": final.get("failed"),
        "average_completion_rate_rps": average_rate,
        "peak_interval_completion_rate_rps": max(interval_rates)
        if interval_rates
        else None,
    }, issues


def summarize_online(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Derive online latency distributions and detect duplicate request indexes."""
    issues = []
    if not records:
        return {"enabled": False, "sample_count": 0}, issues
    indexes = [
        record.get("request_index")
        for record in records
        if isinstance(record.get("request_index"), int)
    ]
    if len(indexes) != len(set(indexes)):
        issues.append("online request indexes contain duplicates")
    successes = [record for record in records if record.get("ok") is True]
    ttft = [
        float(record["ttft_ms"])
        for record in successes
        if isinstance(record.get("ttft_ms"), (int, float))
    ]
    latency = [
        float(record["latency_ms"])
        for record in successes
        if isinstance(record.get("latency_ms"), (int, float))
    ]

    def distribution(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    return {
        "enabled": True,
        "sample_count": len(records),
        "successful_requests": len(successes),
        "failed_requests": len(records) - len(successes),
        "error_rate": (len(records) - len(successes)) / len(records),
        "ttft_ms": distribution(ttft),
        "latency_ms": distribution(latency),
    }, issues


def compile_run(
    experiment_root: Path, run_id: str, output_directory: Path | None = None
) -> Path:
    """Compile a raw run and return the new analysis directory."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise HarnessError(f"invalid run ID: {run_id}")
    raw_directory = experiment_root / "results" / "raw" / run_id
    if not raw_directory.is_dir():
        raise HarnessError(f"raw run does not exist: {raw_directory}")
    metadata = read_json_object(raw_directory / "metadata.json")
    if metadata.get("run_id") != run_id:
        raise HarnessError("metadata run_id does not match the directory")
    analysis_id = f"{run_id}-summary"
    compiled_directory = output_directory or (
        experiment_root / "results" / "compiled" / analysis_id
    )
    compiled_directory.mkdir(parents=True, exist_ok=False)

    progress = read_jsonl(raw_directory / "progress.jsonl")
    online = read_jsonl(raw_directory / "online-requests.jsonl")
    progress_summary, progress_issues = summarize_progress(progress)
    online_summary, online_issues = summarize_online(online)
    source_files = {}
    for filename in (
        "metadata.json",
        "workload-manifest.json",
        "progress.jsonl",
        "online-requests.jsonl",
        "terminal-batch.json",
        "result-validation.json",
        "preflight-summary.json",
    ):
        source_path = raw_directory / filename
        if source_path.is_file():
            source_files[filename] = {
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }

    data_quality_issues = progress_issues + online_issues
    summary = {
        "schema_version": "1.0",
        "analysis_id": analysis_id,
        "created_at": isoformat_utc(),
        "source_run_id": run_id,
        "source_run_status": metadata.get("status"),
        "source_run_kind": metadata.get("kind"),
        "source_control_plane": metadata.get("control_plane"),
        "source_exit_code": metadata.get("exit_code"),
        "batch": progress_summary,
        "online": online_summary,
        "data_quality": {
            "issue_count": len(data_quality_issues),
            "issues": data_quality_issues,
            "missing_progress": not progress,
            "missing_online": not online,
        },
        "source_files": source_files,
    }
    write_json(compiled_directory / "summary.json", summary)
    write_csv(compiled_directory / "progress.csv", PROGRESS_FIELDS, progress)
    if online:
        write_csv(compiled_directory / "online-requests.csv", ONLINE_FIELDS, online)

    fence = chr(96) * 3
    command = f"python3 workloads/compile_run.py --run-id {shlex_quote(run_id)}"
    readme = f"""<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compiled Experiment Run {run_id}

## Purpose

Compile progress and optional online-request observations from raw run
[{run_id}](../../raw/{run_id}/metadata.md).

## Source and Method

- Source run: {run_id}
- Source status: {metadata.get("status")}
- Source run kind: {metadata.get("kind")}
- Source control plane: {(metadata.get("control_plane") or {}).get("mode")}
- Source exit code: {metadata.get("exit_code")}
- Transformation: parse JSONL in file order, validate monotonic progress and
  duplicate online indexes, then calculate nearest-rank p50/p95/p99.
- Missing data: an absent progress or online file produces an explicit empty
  summary and is not imputed.
- Exclusions: none.

## Schemas and Units

- progress.csv records effective and Gateway-reported totals, the total source,
  counts, seconds, and completed requests per second.
- online-requests.csv records milliseconds for queue delay, TTFT, and end-to-end
  latency.
- summary.json contains the derived values and source-file checksums.

## Data Quality

- Issues found: {len(data_quality_issues)}
- Missing progress: {not progress}
- Missing online traffic: {not online}

## Reproduce

Run from the experiment root:

{fence}bash
{command}
{fence}
"""
    write_text(compiled_directory / "README.md", readme)
    return compiled_directory


def shlex_quote(value: str) -> str:
    """Quote one stable identifier without importing shell execution."""
    if re_full_safe(value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def re_full_safe(value: str) -> bool:
    return all(character.isalnum() or character in "._-" for character in value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    args.experiment_root = args.experiment_root.expanduser().resolve()
    if args.output_directory is not None:
        args.output_directory = args.output_directory.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = compile_run(args.experiment_root, args.run_id, args.output_directory)
    except (HarnessError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"compiled results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
