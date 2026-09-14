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
import math
import shutil
import statistics
import sys
import tempfile
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
MAX_JSONL_RECORDS = 100_000


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path} does not contain a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream bounded JSONL while preserving line-number errors."""
    if not path.exists():
        return
    record_count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record_count += 1
            if record_count > MAX_JSONL_RECORDS:
                raise HarnessError(
                    f"{path} exceeds the {MAX_JSONL_RECORDS}-record compile limit"
                )
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError) as error:
                raise HarnessError(
                    f"{path} line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise HarnessError(f"{path} line {line_number} is not an object")
            yield value


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
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Derive batch timing and validate monotonic progress."""
    issues = []
    sample_count = 0
    final: dict[str, Any] | None = None
    previous_elapsed = -1.0
    previous_completed = -1
    totals: set[int] = set()
    interval_rates: list[float] = []
    for index, record in enumerate(records):
        sample_count += 1
        final = record
        elapsed = record.get("elapsed_seconds")
        completed = record.get("completed")
        total = record.get("total")
        rate = record.get("interval_completion_rate_rps")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
        ):
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
        if (
            not isinstance(rate, bool)
            and isinstance(rate, (int, float))
            and math.isfinite(rate)
            and rate >= 0
        ):
            interval_rates.append(float(rate))

    if final is None:
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

    if len(totals) > 1:
        issues.append(f"progress total changed across samples: {sorted(totals)}")

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
        "sample_count": sample_count,
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


def summarize_online(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Derive online latency distributions and detect duplicate request indexes."""
    issues = []
    sample_count = 0
    successful_count = 0
    seen_indexes: set[int] = set()
    duplicate_index = False
    ttft: list[float] = []
    latency: list[float] = []
    for record in records:
        sample_count += 1
        index = record.get("request_index")
        if isinstance(index, int):
            if index in seen_indexes:
                duplicate_index = True
            seen_indexes.add(index)
        if record.get("ok") is not True:
            continue
        successful_count += 1
        ttft_value = record.get("ttft_ms")
        latency_value = record.get("latency_ms")
        if (
            not isinstance(ttft_value, bool)
            and isinstance(ttft_value, (int, float))
            and math.isfinite(ttft_value)
        ):
            ttft.append(float(ttft_value))
        if (
            not isinstance(latency_value, bool)
            and isinstance(latency_value, (int, float))
            and math.isfinite(latency_value)
        ):
            latency.append(float(latency_value))

    if sample_count == 0:
        return {"enabled": False, "sample_count": 0}, issues
    if duplicate_index:
        issues.append("online request indexes contain duplicates")

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
        "sample_count": sample_count,
        "successful_requests": successful_count,
        "failed_requests": sample_count - successful_count,
        "error_rate": (sample_count - successful_count) / sample_count,
        "ttft_ms": distribution(ttft),
        "latency_ms": distribution(latency),
    }, issues


def _write_compiled_artifacts(
    output_directory: Path,
    summary: dict[str, Any],
    progress_path: Path,
    online_path: Path,
    online_sample_count: int,
    readme: str,
) -> None:
    """Write one complete compiled-run tree into a staging directory."""
    write_json(output_directory / "summary.json", summary)
    write_csv(
        output_directory / "progress.csv", PROGRESS_FIELDS, iter_jsonl(progress_path)
    )
    if online_sample_count:
        write_csv(
            output_directory / "online-requests.csv",
            ONLINE_FIELDS,
            iter_jsonl(online_path),
        )
    write_text(output_directory / "README.md", readme)


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

    progress_path = raw_directory / "progress.jsonl"
    online_path = raw_directory / "online-requests.jsonl"
    progress_summary, progress_issues = summarize_progress(iter_jsonl(progress_path))
    online_summary, online_issues = summarize_online(iter_jsonl(online_path))
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
            "missing_progress": progress_summary["sample_count"] == 0,
            "missing_online": online_summary["sample_count"] == 0,
        },
        "source_files": source_files,
    }
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
- Missing progress: {progress_summary["sample_count"] == 0}
- Missing online traffic: {online_summary["sample_count"] == 0}

## Reproduce

Run from the experiment root:

{fence}bash
{command}
{fence}
"""
    compiled_directory.parent.mkdir(parents=True, exist_ok=True)
    if compiled_directory.exists():
        raise FileExistsError(f"output directory already exists: {compiled_directory}")
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{compiled_directory.name}.staging-",
            dir=compiled_directory.parent,
        )
    )
    try:
        _write_compiled_artifacts(
            staging_directory,
            summary,
            progress_path,
            online_path,
            online_summary["sample_count"],
            readme,
        )
        staging_directory.rename(compiled_directory)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors=True)
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
