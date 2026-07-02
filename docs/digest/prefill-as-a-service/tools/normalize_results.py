#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize a validated DYN-3364 AIPerf bundle for publication plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

AGGREGATE_METRICS = {
    "time_to_first_token": ("ms", ("avg", "p50", "p90")),
    "request_latency": ("ms", ("avg", "p50", "p90")),
    "inter_token_latency": ("ms", ("avg", "p50", "p90")),
    "request_throughput": ("requests/sec", ("avg",)),
    "output_token_throughput": ("tokens/sec", ("avg",)),
}

REQUEST_METRICS = {
    "time_to_first_token": "ttft_ms",
    "request_latency": "request_latency_ms",
    "inter_token_latency": "itl_ms",
    "input_sequence_length": "server_input_tokens",
    "output_sequence_length": "server_output_tokens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("aiperf_run_dir", type=Path)
    parser.add_argument("validation_report", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(parameters: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            parameters,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metric_value(record: dict[str, Any], metric: str) -> int | float:
    value = record.get("metrics", {}).get(metric, {}).get("value")
    require(isinstance(value, (int, float)), f"request metric is absent: {metric}")
    return value


def normalize(
    run_root: Path,
    aiperf_run_dir: Path,
    validation_report_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = run_root.resolve()
    aiperf_run_dir = aiperf_run_dir.resolve()
    validation_report_path = validation_report_path.resolve()

    report = read_json(validation_report_path)
    require(
        report.get("valid") is True, "validation report does not mark the run valid"
    )
    checks = report.get("checks")
    require(
        isinstance(checks, list)
        and bool(checks)
        and all(check.get("passed") is True for check in checks),
        "validation report contains a failed or incomplete check",
    )
    reported_run_root = report.get("run_root")
    reported_aiperf_run_dir = report.get("aiperf_run_dir")
    require(
        isinstance(reported_run_root, str)
        and Path(reported_run_root).resolve() == run_root,
        "validation report run_root does not match the supplied run root",
    )
    require(
        isinstance(reported_aiperf_run_dir, str)
        and Path(reported_aiperf_run_dir).resolve() == aiperf_run_dir,
        "validation report AIPerf path does not match the supplied run directory",
    )

    spec_path = run_root / "spec.lock.json"
    spec = read_json(spec_path)
    parameters = spec.get("parameters")
    require(
        isinstance(parameters, dict), "locked specification has no parameters object"
    )
    actual_fingerprint = canonical_fingerprint(parameters)
    require(spec.get("status") == "locked", "specification is not locked")
    require(
        spec.get("fingerprint") == actual_fingerprint,
        "locked specification fingerprint does not match its parameters",
    )

    workload = parameters["workload"]
    expected_repetitions = int(workload["repetitions"])
    expected_requests = int(workload["num_requests"])
    expected_isl = int(workload["input_sequence_length"])
    expected_osl = int(workload["output_sequence_length"])

    report_repetitions = {item["name"]: item for item in report.get("repetitions", [])}
    repetition_dirs = sorted((aiperf_run_dir / "profile_runs").glob("run_*"))
    require(
        len(repetition_dirs) == expected_repetitions,
        "AIPerf repetition count does not match the locked specification",
    )

    normalized_repetitions: list[dict[str, Any]] = []
    normalized_requests: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, str]] = []

    for repetition_index, repetition_dir in enumerate(repetition_dirs, start=1):
        aggregate_path = repetition_dir / "profile_export_aiperf.json"
        records_path = repetition_dir / "profile_export.jsonl"
        aggregate = read_json(aggregate_path)
        records = read_jsonl(records_path)
        validation = report_repetitions.get(repetition_dir.name)
        require(
            validation is not None, f"validation report omits {repetition_dir.name}"
        )
        require(
            len(records) == expected_requests
            and validation.get("raw_record_count") == expected_requests,
            f"request count mismatch in {repetition_dir.name}",
        )

        aggregate_values: dict[str, Any] = {}
        for metric_name, (expected_unit, statistics) in AGGREGATE_METRICS.items():
            metric = aggregate.get(metric_name)
            require(
                isinstance(metric, dict), f"aggregate metric is absent: {metric_name}"
            )
            require(
                metric.get("unit") == expected_unit,
                f"unexpected unit for {metric_name}: {metric.get('unit')}",
            )
            aggregate_values[metric_name] = {
                statistic: metric[statistic] for statistic in statistics
            }
            aggregate_values[metric_name]["unit"] = expected_unit

        normalized_repetitions.append(
            {
                "repetition": repetition_index,
                "name": repetition_dir.name,
                "metrics": aggregate_values,
                "successful_nixl_transfers": validation["successful_transfer_count"],
                "worker_pairs": validation["worker_pairs"],
            }
        )

        seen_request_ids: set[str] = set()
        for record in records:
            request_id = record.get("metadata", {}).get("x_request_id")
            require(isinstance(request_id, str) and request_id, "request ID is absent")
            require(
                request_id not in seen_request_ids,
                f"duplicate request ID: {request_id}",
            )
            seen_request_ids.add(request_id)
            normalized_request: dict[str, Any] = {
                "repetition": repetition_index,
                "request_id": request_id,
            }
            for source_name, output_name in REQUEST_METRICS.items():
                normalized_request[output_name] = metric_value(record, source_name)
            require(
                normalized_request["server_input_tokens"] > 0,
                f"request {request_id} has no server-reported input length",
            )
            require(
                normalized_request["server_output_tokens"] == expected_osl,
                f"request {request_id} has an unexpected output length",
            )
            normalized_requests.append(normalized_request)

        for path in (aggregate_path, records_path):
            source_artifacts.append(
                {
                    "path": str(path.relative_to(run_root)),
                    "sha256": sha256_file(path),
                }
            )

    normalized = {
        "schema_version": 1,
        "run_id": aiperf_run_dir.name,
        "spec_fingerprint": actual_fingerprint,
        "aiperf_spec": spec.get("aiperf_spec"),
        "validation_report_sha256": sha256_file(validation_report_path),
        "locked_workload": {
            "repetitions": expected_repetitions,
            "requests_per_repetition": expected_requests,
            "configured_input_sequence_length": expected_isl,
            "configured_output_sequence_length": expected_osl,
        },
        "repetitions": normalized_repetitions,
        "requests": normalized_requests,
        "source_artifacts": source_artifacts,
    }
    return normalized, normalized_requests


def write_outputs(
    normalized: dict[str, Any],
    requests: list[dict[str, Any]],
    json_output: Path,
    csv_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    json_temporary = json_output.with_suffix(json_output.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_output)

    fieldnames = ["repetition", "request_id", *REQUEST_METRICS.values()]
    csv_temporary = csv_output.with_suffix(csv_output.suffix + ".tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(requests)
    csv_temporary.replace(csv_output)


def main() -> int:
    args = parse_args()
    normalized, requests = normalize(
        args.run_root,
        args.aiperf_run_dir,
        args.validation_report,
    )
    write_outputs(
        normalized,
        requests,
        args.json_output,
        args.csv_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
