#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# On-CPU profile of a target process (frontend or mocker) under load.
# Non-root (needs perf_event_paranoid <= 1; -1 is ideal). Topology must be up.
# Also samples the target's cores so you can see if it saturates.
#
#   DYN_REPO=/path bash profile_oncpu.sh --frontend [--conc 2048]
#   DYN_REPO=/path bash profile_oncpu.sh --mocker
#   DYN_REPO=/path bash profile_oncpu.sh --pid <PID> --cores 0-3
set -uo pipefail
cd "$(dirname "$0")"; source ./env.sh

CONC=2048; LOAD_SECS=150; SETTLE=40; CAP_SECS=40; PID=""; CORES=""; TAG="proc"
while [[ $# -gt 0 ]]; do case $1 in
  --frontend) PID="$(cat "$LOG_DIR/frontend.pid")"; CORES="$FRONTEND_CORES"; TAG="frontend"; shift;;
  --mocker)   PID="$(pgrep -f 'python.*dynamo.mocker'|head -1)"; CORES="$OTHER_CORES"; TAG="mocker"; shift;;
  --pid)      PID="$2"; shift 2;;
  --cores)    CORES="$2"; shift 2;;
  --conc)     CONC="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null || { echo "ERROR: target PID '$PID' not running."; exit 1; }

expand_cpu_list() {
  local spec="$1"
  local part start end cpu
  local out=()
  IFS=',' read -ra parts <<< "$spec"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start <= end )); then
        for ((cpu=start; cpu<=end; cpu++)); do out+=("$cpu"); done
      else
        for ((cpu=start; cpu>=end; cpu--)); do out+=("$cpu"); done
      fi
    else
      out+=("$part")
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_DIR/../profiling/oncpu-$TAG-$STAMP"; mkdir -p "$OUT"
echo "[oncpu] target=$TAG PID=$PID cores=$CORES conc=$CONC out=$OUT"

taskset -c "$OTHER_CORES" "$AIPERF" profile --model "$MODEL" --tokenizer "$MODEL" \
  --url "http://localhost:${HTTP_PORT}" --endpoint-type chat --streaming \
  --shared-system-prompt-length 48000 --user-context-prompt-length 12000 \
  --num-dataset-entries 1024 --output-tokens-mean 500 --conversation-turn-mean 4 \
  --concurrency "$CONC" --benchmark-duration "$LOAD_SECS" --warmup-request-count 256 \
  --extra-inputs "ignore_eos:true" --artifact-dir "$OUT/load_artifacts" \
  > "$OUT/load_aiperf.log" 2>&1 &
LOAD_PID=$!
trap 'kill "$LOAD_PID" 2>/dev/null; pkill -f "aiperf profile" 2>/dev/null' EXIT
echo "[load] settling ${SETTLE}s ..."; sleep "$SETTLE"

# per-core utilization of the target's cores + per-process CPU
if [[ -n "$CORES" ]]; then
  MPSTAT_CORES="$(expand_cpu_list "$CORES")"
  mpstat -P "$MPSTAT_CORES" "$CAP_SECS" 1 > "$OUT/mpstat.txt" 2>/dev/null &
fi
pidstat -p "$PID" 2 > "$OUT/pidstat.txt" 2>/dev/null & PS1=$!

echo "[perf] on-CPU record ${CAP_SECS}s ..."
# DWARF call graph: the release .so has no frame pointers, so -g (FP) would
# truncate stacks. -F 99 keeps overhead low; raise to 199/499 for more detail.
perf record -F 99 --call-graph dwarf -p "$PID" -o "$OUT/oncpu.data" -- sleep "$CAP_SECS" 2>"$OUT/perf.err" || echo "perf failed"
kill "$PS1" 2>/dev/null || true

echo "[fold] flamegraph ..."
perf script -i "$OUT/oncpu.data" 2>/dev/null | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" > "$OUT/oncpu.folded" 2>/dev/null
"$FLAMEGRAPH_DIR/flamegraph.pl" --title "on-CPU $TAG (c=$CONC)" "$OUT/oncpu.folded" > "$OUT/oncpu.svg" 2>/dev/null

kill "$LOAD_PID" 2>/dev/null; pkill -f "aiperf profile" 2>/dev/null; trap - EXIT
echo "[done] $OUT"
echo "--- target cores busy (mpstat) ---"
[[ -f "$OUT/mpstat.txt" ]] && awk '$NF ~ /^[0-9.]+$/ && $3 ~ /^[0-9]+$/ {b=100-$NF; s+=b; n++} END{if(n)printf "mean %.1f%%/core\n",s/n}' "$OUT/mpstat.txt"
echo "--- analyze with:  python3 analyze_folded.py $OUT/oncpu.folded ---"
