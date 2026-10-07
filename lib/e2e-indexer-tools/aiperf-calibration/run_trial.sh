#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One calibration trial: stub server + K AIPerf instances + CPU sampler. Run under node-gate.sh.
# Env:
#   NAME            trial name (results go to $R/results/$NAME)
#   STUB_CPUS       comma list, one stub process per CPU
#   STUB_ITL_MS     stub inter-token delay (default 0 = whole stream in one write)
#   AIPERF_CPUS     ';'-separated CPU lists, one AIPerf instance per entry (taskset -c)
#   AIPERF_ARGS     arguments shared by every instance (url/model/tokenizer/artifact-dir are added)
#   AIPERF_ENV      extra env assignments for AIPerf (space-separated KEY=VAL)
#   OFFERED         offered req/s per instance (metadata only)
set -uo pipefail
: "${SLURM_JOB_ID:?run inside the hold}" "${NAME:?}" "${STUB_CPUS:?}" "${AIPERF_CPUS:?}" "${AIPERF_ARGS:?}"
R=${CALIB_ROOT:?set CALIB_ROOT to the shared calibration directory}
L=${CALIB_LOCAL:-/tmp/aiperf-calib-$SLURM_JOB_ID}
S=$R/scripts
PY=$L/venv/bin/python
OUT=$L/runs/$NAME
rm -rf "$OUT"
mkdir -p "$OUT"
export HF_HOME=$L/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
export AIPERF_DATASET_MMAP_CACHE_DIR=$L/mmap XDG_CACHE_HOME=$L/xdg
PORT=${PORT:-18000}
pids=()
cleanup() {
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 1
  for p in "${pids[@]}"; do kill -KILL "$p" 2>/dev/null; done
}
trap cleanup EXIT

$PY "$S/stub_server.py" --port "$PORT" --cpus "$STUB_CPUS" --itl-ms "${STUB_ITL_MS:-0}" \
  --report-s 2 >"$OUT/stub.log" 2>&1 &
stub=$!
pids+=("$stub")
for _ in $(seq 50); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 0.2
done

IFS=';' read -r -a cpusets <<<"$AIPERF_CPUS"
roots=("stub=$stub")
apids=()
t0=$(date +%s.%3N)
for i in "${!cpusets[@]}"; do
  mkdir -p "$OUT/i$i"
  # shellcheck disable=SC2086
  env ${AIPERF_ENV:-} taskset -c "${cpusets[$i]}" "$L/venv/bin/aiperf" profile \
    --url "127.0.0.1:$PORT" --model stub --tokenizer Qwen/Qwen3-0.6B --ui none \
    --artifact-dir "$OUT/i$i/art" $AIPERF_ARGS >"$OUT/i$i/aiperf.log" 2>&1 &
  apids+=("$!")
  pids+=("$!")
  roots+=("aiperf$i=$!")
done
taskset -c "${SAMPLER_CPU:-127}" $PY "$S/cpu_sampler.py" --out "$OUT/cpu.jsonl" --interval 1 "${roots[@]}" &
sampler=$!
pids+=("$sampler")
rc=0
for p in "${apids[@]}"; do
  wait "$p" || rc=$?
done
t1=$(date +%s.%3N)
kill -TERM "$sampler" "$stub" 2>/dev/null
wait "$sampler" "$stub" 2>/dev/null
printf '{"name":"%s","t0":%s,"t1":%s,"rc":%s,"stub_cpus":"%s","aiperf_cpus":"%s","offered":"%s","stub_itl_ms":"%s","aiperf_args":"%s","aiperf_env":"%s"}\n' \
  "$NAME" "$t0" "$t1" "$rc" "$STUB_CPUS" "$AIPERF_CPUS" "${OFFERED:-}" "${STUB_ITL_MS:-0}" "$AIPERF_ARGS" "${AIPERF_ENV:-}" >"$OUT/meta.json"
$PY "$S/summarize.py" "$OUT" | tee "$OUT/summary.txt"
mkdir -p "$R/results/$NAME"
# Keep summaries, aggregate exports and logs; per-request jsonl stays node-local unless KEEP_RECORDS=1.
rsync -a --exclude 'profile_export.jsonl' --exclude '*_raw.jsonl' --exclude 'inputs.json' "$OUT/" "$R/results/$NAME/"
[[ "${KEEP_RECORDS:-0}" == 1 ]] && rsync -a "$OUT/" "$R/results/$NAME/"
exit "$rc"
