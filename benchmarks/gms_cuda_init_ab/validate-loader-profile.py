# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Runtime or Driver CUDA profile records from the GMS loader."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROFILE_PREFIX = "GMS_SNAPSHOT_PROFILE "
DRIVER_COUNTS = {
    "loader_cu_init": 1,
    "cu_device_get": 8,
    "primary_context_retain": 8,
    "loader_cu_ctx_set_current": 8,
    "first_h2d_submission": 8,
    "all_device_primary_context_retain_envelope": 1,
    "all_device_cu_ctx_set_current_envelope": 1,
    "all_device_driver_initialization": 1,
    "loader_cuda_initialization_total": 1,
    "primary_context_release": 1,
}
RUNTIME_COUNTS = {
    "cuda_set_device": 8,
    "current_context_query": 8,
    "all_device_cuda_initialization": 1,
}
RUNTIME_ONLY_PHASES = {
    "client_cu_init",
    "first_claimed_cuda_set_device",
    "staging_worker_cuda_set_device",
    "transfer_worker_cuda_set_device",
}


def records_from_log(log_path: str) -> list[dict[str, object]]:
    records = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if PROFILE_PREFIX not in line:
            continue
        record = json.loads(line.split(PROFILE_PREFIX, 1)[1])
        if record.get("component") == "loader":
            records.append(record)
    return records


def validate_counts(
    counts: Counter[str],
    expected: dict[str, int],
    errors: list[str],
) -> None:
    for phase, expected_count in expected.items():
        observed = counts[phase]
        if observed != expected_count:
            errors.append(
                f"{phase} count mismatch: got={observed} expected={expected_count}"
            )


def main(log_path: str, variant: str) -> int:
    records = records_from_log(log_path)
    counts = Counter(str(record.get("phase")) for record in records)
    errors = []
    if variant == "c":
        validate_counts(counts, DRIVER_COUNTS, errors)
        forbidden = sorted(
            phase
            for phase in RUNTIME_ONLY_PHASES | set(RUNTIME_COUNTS)
            if counts[phase]
        )
        if forbidden:
            errors.append(f"Driver mode reached Runtime-only phases: {forbidden}")
        driver_aggregate_phases = {
            str(record.get("phase"))
            for record in records
            if record.get("cuda_api") == "driver"
        }
        required_driver_aggregates = {
            "cuda_host_register",
            "cuda_stream_create",
            "cuda_event_create",
            "staging_worker_cu_ctx_set_current",
            "transfer_worker_cu_ctx_set_current",
        }
        missing = sorted(required_driver_aggregates - driver_aggregate_phases)
        if missing:
            errors.append(f"missing Driver aggregate phases: {missing}")
        runtime_records = [
            record
            for record in records
            if str(record.get("api", "")).startswith("cuda")
            or record.get("cuda_api") == "runtime"
        ]
        if runtime_records:
            errors.append(
                f"Driver mode emitted {len(runtime_records)} Runtime API record(s)"
            )
    else:
        validate_counts(counts, RUNTIME_COUNTS, errors)
        if counts["first_claimed_cuda_set_device"] != 1:
            errors.append(
                "Runtime mode first_claimed_cuda_set_device count mismatch: "
                f"got={counts['first_claimed_cuda_set_device']} expected=1"
            )
        unexpected = sorted(phase for phase in DRIVER_COUNTS if counts[phase])
        if unexpected:
            errors.append(f"Runtime mode emitted Driver-only phases: {unexpected}")

    failed_required = [
        record
        for record in records
        if record.get("status") == "error"
        and str(record.get("phase")) in DRIVER_COUNTS | RUNTIME_COUNTS
    ]
    if failed_required:
        errors.append(f"required CUDA profile phases failed: {failed_required}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"validated loader variant={variant.upper()} records={len(records)} "
        f"phases={len(counts)}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in {"a", "b", "c"}:
        raise SystemExit(f"usage: {sys.argv[0]} LOADER_LOG {{a|b|c}}")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
