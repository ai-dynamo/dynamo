#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 EVIDENCE_DIRECTORY" >&2
    exit 2
fi

ART=$1
COUNTS=${COUNTS:-585,64,16,4,1}
REPEATS=${REPEATS:-3}
SCRIPT=${SCRIPT:-/opt/gms-experiment/vmm-slab-probe.py}
SIZES=/opt/gms-experiment/allocation-sizes.json
PIDS=()

if [[ ! -d "$ART" ]]; then
    mkdir -- "$ART"
elif [[ -n "$(find "$ART" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing non-empty evidence directory: $ART" >&2
    exit 1
fi

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap cleanup EXIT INT TERM

IFS=, read -r -a counts <<< "$COUNTS"
for count in "${counts[@]}"; do
    for repeat in $(seq 1 "$REPEATS"); do
        run="$ART/count-$count-repeat-$repeat"
        mkdir "$run"
        nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
            > "$run/memory-before.csv"
        PIDS=()
        for device in $(seq 0 7); do
            uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader |
                sed -n "$((device + 1))p" | tr -d ' ')
            socket_path="/tmp/gms-vmm-probe-$count-$repeat-$device.sock"
            CUDA_VISIBLE_DEVICES="$uuid" python3 "$SCRIPT" exporter \
                --device "$device" \
                --physical-count "$count" \
                --sizes "$SIZES" \
                --socket "$socket_path" \
                --result "$run/exporter-$device.json" &
            PIDS+=("$!")
        done
        for device in $(seq 0 7); do
            uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader |
                sed -n "$((device + 1))p" | tr -d ' ')
            socket_path="/tmp/gms-vmm-probe-$count-$repeat-$device.sock"
            CUDA_VISIBLE_DEVICES="$uuid" python3 "$SCRIPT" importer \
                --device "$device" \
                --physical-count "$count" \
                --sizes "$SIZES" \
                --socket "$socket_path" \
                --result "$run/importer-$device.json" &
            PIDS+=("$!")
        done
        status=0
        for pid in "${PIDS[@]}"; do
            wait "$pid" || status=$?
        done
        PIDS=()
        [[ "$status" -eq 0 ]]
        nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
            > "$run/memory-after.csv"
        python3 - "$run/memory-before.csv" "$run/memory-after.csv" <<'PY'
import sys
from pathlib import Path


def load(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        uuid, used = [field.strip() for field in line.split(",")]
        records[uuid] = int(used)
    return records


before = load(sys.argv[1])
after = load(sys.argv[2])
if len(before) != 8 or set(after) != set(before):
    raise SystemExit(f"GPU memory UUID mismatch: before={before} after={after}")
regressions = {
    uuid: (before[uuid], after[uuid])
    for uuid in before
    if after[uuid] > before[uuid]
}
if regressions:
    raise SystemExit(f"GPU memory did not return to baseline: {regressions}")
PY
        python3 - "$run" "$count" "$repeat" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
count = int(sys.argv[2])
repeat = int(sys.argv[3])
records = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(run.glob("*.json"))
]
if len(records) != 16:
    raise SystemExit(f"expected 16 records, got {len(records)}")
for record in records:
    if record["physical_count"] != count or record["bytes"] != 58_745_421_824:
        raise SystemExit(f"invalid probe record: {record}")
phases = sorted(
    {phase for record in records for phase in record["phases"]}
)
summary = {
    "physical_count": count,
    "repeat": repeat,
    "devices": 8,
    "bytes_per_device": 58_745_421_824,
    "phases": {},
}
for phase in phases:
    phase_records = [
        record["phases"][phase]
        for record in records
        if phase in record["phases"]
    ]
    summary["phases"][phase] = {
        "process_count": len(phase_records),
        "duration_sum_ns": sum(record["duration_ns"] for record in phase_records),
        "wall_start_ns": min(record["wall_start_ns"] for record in phase_records),
        "wall_end_ns": max(record["wall_end_ns"] for record in phase_records),
    }
    summary["phases"][phase]["wall_envelope_ns"] = (
        summary["phases"][phase]["wall_end_ns"]
        - summary["phases"][phase]["wall_start_ns"]
    )
(run / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
    done
done

find "$ART" -type f ! -path "$ART/SHA256SUMS" -print0 |
    sort -z |
    xargs -0 sha256sum > "$ART/SHA256SUMS"
sha256sum -c "$ART/SHA256SUMS"
