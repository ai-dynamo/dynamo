# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate physical GPU attribution in merged GMS server logs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROFILE_PREFIX = "GMS_SNAPSHOT_PROFILE "
PHASES = {"server_cu_init", "allocation_manager_ready", "socket_ready"}
ALLOCATION_COUNT = 4680
ALLOCATION_BYTES = 469_963_374_592
LAUNCH = re.compile(
    r"Started GMS device=\d+ physical_uuid=(GPU-[^ ]+) " r"child_device=0 pid=(\d+)"
)


def main(log_path: str, expected_path: str, variant: str) -> int:
    text = Path(log_path).read_text(encoding="utf-8")
    expected = set(Path(expected_path).read_text(encoding="utf-8").splitlines())
    launches = {uuid: int(pid) for uuid, pid in LAUNCH.findall(text)}
    records = []
    for line in text.splitlines():
        if PROFILE_PREFIX not in line:
            continue
        record = json.loads(line.split(PROFILE_PREFIX, 1)[1])
        if record.get("component") == "server" and record.get("phase") in PHASES:
            records.append(record)

    errors = []
    isolated_variants = {"b", "c", "m", "p", "mp"}
    if variant in isolated_variants and set(launches) != expected:
        errors.append(
            f"isolated launch UUIDs mismatch: got={sorted(launches)} "
            f"expected={sorted(expected)}"
        )
    for phase in PHASES:
        phase_records = [record for record in records if record["phase"] == phase]
        phase_uuids = {record.get("physical_uuid") for record in phase_records}
        if phase_uuids != expected:
            errors.append(
                f"{phase} UUIDs mismatch: got={sorted(phase_uuids)} "
                f"expected={sorted(expected)}"
            )
        for record in phase_records:
            if variant in isolated_variants and record.get("device") != 0:
                errors.append(f"{phase} has nonzero child device: {record}")
            if not isinstance(record.get("pid"), int) or record["pid"] <= 0:
                errors.append(f"{phase} has invalid PID: {record}")
            uuid = record.get("physical_uuid")
            if variant in isolated_variants and launches.get(uuid) != record.get("pid"):
                errors.append(f"{phase} PID does not match launch: {record}")

    allocation_records = {}
    for line in text.splitlines():
        if PROFILE_PREFIX not in line:
            continue
        record = json.loads(line.split(PROFILE_PREFIX, 1)[1])
        phase = record.get("phase")
        if record.get("component") == "server" and phase in {
            "server_cu_mem_create",
            "server_initial_cuda_export",
            "server_export_fd_dup",
        }:
            allocation_records.setdefault(phase, []).append(record)
    for phase in (
        "server_cu_mem_create",
        "server_initial_cuda_export",
        "server_export_fd_dup",
    ):
        phase_records = allocation_records.get(phase, [])
        if sum(int(record.get("count", 0)) for record in phase_records) != (
            ALLOCATION_COUNT
        ):
            errors.append(f"{phase} allocation count mismatch")
        expected_bytes = 0 if phase == "server_export_fd_dup" else ALLOCATION_BYTES
        if (
            sum(int(record.get("bytes", 0)) for record in phase_records)
            != expected_bytes
        ):
            errors.append(f"{phase} allocation byte count mismatch")

    readiness_device = "0" if variant in isolated_variants else r"\d+"
    for uuid in expected:
        if not re.search(
            rf"Server started: .*device={readiness_device} "
            rf"physical_uuid={re.escape(uuid)} pid=\d+",
            text,
        ):
            errors.append(f"missing attributed readiness log for {uuid}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"validated variant={variant.upper()} uuids={len(expected)} "
        f"records={len(records)}"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in {"a", "b", "c", "m", "p", "mp"}:
        raise SystemExit(
            f"usage: {sys.argv[0]} SERVER_LOG EXPECTED_UUIDS {{a|b|c|m|p|mp}}"
        )
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
