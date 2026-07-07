#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <binary> <corpus.msgpack> <corpus-sha256> <measured-code-sha> <run-root>" >&2
  exit 2
fi

binary=$1
corpus=$2
corpus_sha=$3
measured_code_sha=$4
run_root=$5
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
python=${PYTHON:-"$repo_root/.venv/bin/python"}
event_threads=${EVENT_THREADS:-8}
query_concurrency=${QUERY_CONCURRENCY:-16}

[[ -x "$binary" ]] || { echo "benchmark binary is not executable: $binary" >&2; exit 2; }
[[ -x "$python" ]] || { echo "project Python is not executable: $python" >&2; exit 2; }
[[ -f "$corpus" ]] || { echo "corpus does not exist: $corpus" >&2; exit 2; }

actual_sha=$(sha256sum "$corpus" | awk '{print $1}')
[[ "$actual_sha" == "$corpus_sha" ]] || {
  echo "corpus SHA mismatch: expected $corpus_sha, got $actual_sha" >&2
  exit 2
}

mkdir -p "$run_root/trials" "$run_root/logs" "$run_root/aggregate"
corpus_manifest="${corpus%.*}.manifest.json"
[[ -f "$corpus_manifest" ]] || { echo "corpus manifest does not exist: $corpus_manifest" >&2; exit 2; }
cp "$corpus_manifest" "$run_root/corpus_manifest.json"
[[ -n ${HARDWARE_MANIFEST:-} ]] || { echo "HARDWARE_MANIFEST is required" >&2; exit 2; }
[[ -f "$HARDWARE_MANIFEST" ]] || { echo "hardware manifest does not exist: $HARDWARE_MANIFEST" >&2; exit 2; }
cp "$HARDWARE_MANIFEST" "$run_root/hardware.json"
manifest_corpus_sha=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_sha256"])' "$corpus_manifest")
[[ "$manifest_corpus_sha" == "$corpus_sha" ]] || {
  echo "manifest corpus SHA mismatch: expected $corpus_sha, got $manifest_corpus_sha" >&2
  exit 2
}
export EXPECTED_CPU_BINDING
EXPECTED_CPU_BINDING=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cpu_binding"])' "$HARDWARE_MANIFEST")
[[ -n "$EXPECTED_CPU_BINDING" ]] || { echo "hardware manifest has no CPU binding" >&2; exit 2; }

prefix=()
if [[ -n ${RUN_PREFIX:-} ]]; then
  read -r -a prefix <<<"$RUN_PREFIX"
fi

run_one() {
  local phase=$1
  local repetition=$2
  local backend=$3
  local window_ms=$4
  local window_tag=${window_ms//./p}
  local stem
  stem=$(printf '%s_rep%02d_%s_%sms' "$phase" "$repetition" "$backend" "$window_tag")
  local output="$run_root/trials/$stem.json"
  local log="$run_root/logs/$stem.log"
  if [[ -s "$output" ]]; then
    if "$python" - "$output" "$phase" "$repetition" "$backend" "$window_ms" "$corpus_sha" "$measured_code_sha" <<'PY'
import json
import math
import sys

path, phase, repetition, backend, window_ms, corpus_sha, code_sha = sys.argv[1:]
row = json.load(open(path))
valid = (
    row.get("phase") == phase
    and int(row.get("repetition", -1)) == int(repetition)
    and row.get("backend") == backend
    and math.isclose(float(row.get("replay_window_ms", -1)), float(window_ms), rel_tol=1e-9)
    and row.get("corpus_sha256") == corpus_sha
    and row.get("measured_code_sha") == code_sha
)
raise SystemExit(0 if valid else 1)
PY
    then
      echo "preserving verified existing trial $output"
      return
    fi
    echo "existing trial does not match this run: $output" >&2
    exit 1
  fi
  "${prefix[@]}" "$binary" run-cell \
    --corpus "$corpus" \
    --expected-corpus-sha256 "$corpus_sha" \
    --backend "$backend" \
    --replay-window-ms "$window_ms" \
    --repetition "$repetition" \
    --phase "$phase" \
    --measured-code-sha "$measured_code_sha" \
    --output "$output" \
    >"$log" 2>&1
  [[ -s "$output" ]] || { echo "trial did not produce $output" >&2; exit 1; }
}

backends=(crtc ckf-native ckf-transposed)
windows=(24000 12000 6000 3000 1500 750)

for repetition_index in 0 1 2 3 4; do
  repetition=$((repetition_index + 1))
  for window_index in 0 1 2 3 4 5; do
    rotated_window_index=$(((window_index + repetition_index) % 6))
    window=${windows[$rotated_window_index]}
    for backend_index in 0 1 2; do
      rotated_backend_index=$(((backend_index + repetition_index + window_index) % 3))
      backend=${backends[$rotated_backend_index]}
      run_one capacity "$repetition" "$backend" "$window"
    done
  done
done

capacity_count=$(find "$run_root/trials" -maxdepth 1 -name 'capacity_*.json' | wc -l | tr -d ' ')
[[ "$capacity_count" == 90 ]] || { echo "expected 90 capacity trials, found $capacity_count" >&2; exit 1; }

"$python" "$script_dir/analyze.py" \
  --stage capacity \
  --results-dir "$run_root/trials" \
  --output-dir "$run_root/aggregate" \
  --event-threads "$event_threads" \
  --query-concurrency "$query_concurrency"

iso_window_ms=$(
  "$python" - "$run_root/aggregate/iso_plan.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["iso_window_ms"])
PY
)

for repetition_index in 0 1 2 3 4; do
  repetition=$((repetition_index + 1))
  for backend_index in 0 1 2; do
    rotated_backend_index=$(((backend_index + repetition_index) % 3))
    run_one iso "$repetition" "${backends[$rotated_backend_index]}" "$iso_window_ms"
  done
done

"$python" "$script_dir/analyze.py" \
  --stage iso-check \
  --results-dir "$run_root/trials" \
  --output-dir "$run_root/aggregate" \
  --event-threads "$event_threads" \
  --query-concurrency "$query_concurrency"

retry_required=$(
  "$python" - "$run_root/aggregate/iso_check.json" <<'PY'
import json
import sys
print("yes" if json.load(open(sys.argv[1]))["retry_required"] else "no")
PY
)

if [[ "$retry_required" == yes ]]; then
  retry_window_ms=$(
    "$python" - "$run_root/aggregate/iso_check.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["retry_iso_window_ms"])
PY
  )
  for repetition_index in 0 1 2 3 4; do
    repetition=$((repetition_index + 1))
    for backend_index in 0 1 2; do
      rotated_backend_index=$(((backend_index + repetition_index) % 3))
      run_one iso-retry "$repetition" "${backends[$rotated_backend_index]}" "$retry_window_ms"
    done
  done
fi

resident_image="${corpus%.*}.resident.msgpack"
[[ -f "$resident_image" ]] || {
  echo "resident image does not exist: $resident_image" >&2
  exit 2
}
resident_sha=$(sha256sum "$resident_image" | awk '{print $1}')
manifest_resident_sha=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["resident_sha256"])' "$corpus_manifest")
[[ "$manifest_resident_sha" == "$resident_sha" ]] || {
  echo "manifest resident SHA mismatch: expected $manifest_resident_sha, got $resident_sha" >&2
  exit 2
}
for memory_mode in crtc ckf-native ckf-transposed relay-producer; do
  output="$run_root/trials/memory_${memory_mode}.json"
  log="$run_root/logs/memory_${memory_mode}.log"
  if [[ -s "$output" ]]; then
    if "$python" - "$output" "$memory_mode" "$resident_sha" "$measured_code_sha" <<'PY'
import json
import sys

path, mode, resident_sha, code_sha = sys.argv[1:]
row = json.load(open(path))
valid = (
    row.get("mode") == mode
    and row.get("resident_image_sha256") == resident_sha
    and row.get("measured_code_sha") == code_sha
)
raise SystemExit(0 if valid else 1)
PY
    then
      echo "preserving verified existing memory result $output"
      continue
    fi
    echo "existing memory result does not match this run: $output" >&2
    exit 1
  fi
  "${prefix[@]}" "$binary" memory \
    --resident-image "$resident_image" \
    --mode "$memory_mode" \
    --measured-code-sha "$measured_code_sha" \
    --output "$output" \
    >"$log" 2>&1
done

"$python" "$script_dir/analyze.py" \
  --stage final \
  --results-dir "$run_root/trials" \
  --output-dir "$run_root/aggregate" \
  --event-threads "$event_threads" \
  --query-concurrency "$query_concurrency"

iso_count=$(find "$run_root/trials" -maxdepth 1 -name 'iso_rep*.json' | wc -l | tr -d ' ')
[[ "$iso_count" == 15 ]] || { echo "expected 15 initial iso trials, found $iso_count" >&2; exit 1; }
memory_count=$(find "$run_root/trials" -maxdepth 1 -name 'memory_*.json' | wc -l | tr -d ' ')
[[ "$memory_count" == 4 ]] || { echo "expected 4 memory trials, found $memory_count" >&2; exit 1; }

find "$run_root/trials" "$run_root/logs" "$run_root/aggregate" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"$run_root/raw_checksums.sha256"
for manifest in "$run_root/corpus_manifest.json" "$run_root/hardware.json"; do
  [[ -f "$manifest" ]] && sha256sum "$manifest" >>"$run_root/raw_checksums.sha256"
done

echo "capacity_trials=$capacity_count"
echo "iso_trials=$iso_count"
echo "memory_trials=$memory_count"
echo "retry_required=$retry_required"
echo "aggregate=$run_root/aggregate/aggregate_results.json"
