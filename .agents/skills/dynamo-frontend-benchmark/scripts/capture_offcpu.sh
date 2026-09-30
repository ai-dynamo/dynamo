#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Off-CPU profile of a target process (frontend or mocker) under load.
# Shows what blocked threads are WAITING on (futex/lock, epoll/network, park).
#
# MUST run as root: sched tracepoints + BPF are root-only even with
# perf_event_paranoid=-1 (tracefs event files are root-readable only).
#
#   sudo DYN_REPO=/path/to/dynamo bash capture_offcpu.sh --pid <PID> [--conc 2048]
#   # or target the frontend automatically:
#   sudo DYN_REPO=/path/to/dynamo bash capture_offcpu.sh --frontend
#
# Pre-req: topology already up (start.sh). This only drives load + captures.
set -uo pipefail
cd "$(dirname "$0")"; source ./env.sh

CONC=2048; LOAD_SECS=200; SETTLE=45; CAP_SECS=45; MIN_US=1000; PID=""; TAG="proc"
while [[ $# -gt 0 ]]; do case $1 in
  --pid) PID="$2"; shift 2;;
  --frontend) PID="$(cat "$LOG_DIR/frontend.pid")"; TAG="frontend"; shift;;
  --mocker) PID="$(pgrep -f 'python.*dynamo.mocker'|head -1)"; TAG="mocker"; shift;;
  --conc) CONC="$2"; shift 2;;
  --cap) CAP_SECS="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done

[[ $EUID -eq 0 ]] || { echo "ERROR: run with sudo (sched tracepoints + BPF need root)."; exit 1; }
[[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null || { echo "ERROR: target PID '$PID' not running."; exit 1; }
command -v setsid >/dev/null 2>&1 || { echo "ERROR: setsid is required to isolate the aiperf load process group."; exit 1; }
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_DIR/../profiling/offcpu-$TAG-$STAMP"; mkdir -p "$OUT"
echo "[offcpu] target=$TAG PID=$PID conc=$CONC out=$OUT"

cleanup_load() {
  if [[ -n "${LOAD_PGID:-}" ]]; then
    kill -- "-$LOAD_PGID" 2>/dev/null || true
  elif [[ -n "${LOAD_PID:-}" ]]; then
    kill "$LOAD_PID" 2>/dev/null || true
  fi
}

# Drive load (time-based; small dataset = fast client-side gen).
setsid taskset -c "$OTHER_CORES" "$AIPERF" profile --model "$MODEL" --tokenizer "$MODEL" \
  --url "http://localhost:${HTTP_PORT}" --endpoint-type chat --streaming \
  --shared-system-prompt-length 48000 --user-context-prompt-length 12000 \
  --num-dataset-entries 1024 --output-tokens-mean 500 --conversation-turn-mean 4 \
  --concurrency "$CONC" --benchmark-duration "$LOAD_SECS" --warmup-request-count 256 \
  --extra-inputs "ignore_eos:true" --artifact-dir "$OUT/load_artifacts" \
  > "$OUT/load_aiperf.log" 2>&1 &
LOAD_PID=$!
LOAD_PGID="$(ps -o pgid= -p "$LOAD_PID" 2>/dev/null | tr -d '[:space:]')"
LOAD_PGID="${LOAD_PGID:-$LOAD_PID}"
trap cleanup_load EXIT
echo "[load] settling ${SETTLE}s ..."; sleep "$SETTLE"

# 1) bcc offcputime: duration-weighted, user+kernel, folded directly.
echo "[cap] offcputime-bpfcc ${CAP_SECS}s ..."
offcputime-bpfcc -df -d "$CAP_SECS" -p "$PID" -m "$MIN_US" > "$OUT/offcpu_bcc.folded" 2>"$OUT/offcpu_bcc.err" \
  || echo "  (bcc failed; see offcpu_bcc.err)"

# 2) perf sched_switch + DWARF: reliable Rust user stacks (release .so has no frame pointers).
echo "[cap] perf sched_switch --call-graph dwarf ${CAP_SECS}s ..."
perf record -e sched:sched_switch --call-graph dwarf -p "$PID" \
  -o "$OUT/sched.data" -- sleep "$CAP_SECS" 2>"$OUT/perf_record.err" \
  || echo "  (perf failed; see perf_record.err)"

# 3) flamegraph from the bcc folded stacks.
if [[ -s "$OUT/offcpu_bcc.folded" && -x "$FLAMEGRAPH_DIR/flamegraph.pl" ]]; then
  "$FLAMEGRAPH_DIR/flamegraph.pl" --title "off-CPU $TAG (c=$CONC)" --countname us --color io \
    "$OUT/offcpu_bcc.folded" > "$OUT/offcpu_bcc.svg" 2>/dev/null || true
fi

cleanup_load; trap - EXIT
# Hand artifacts back to the invoking (non-root) user so analysis can read them.
chown -R "$(stat -c %U "$RESULTS_DIR")":"$(stat -c %G "$RESULTS_DIR")" "$OUT" 2>/dev/null || true
echo "[done] $OUT  (offcpu_bcc.folded / .svg ; sched.data for perf-DWARF)"
echo "REMINDER: aiperf ran as ROOT; if orphans linger, 'sudo pkill -9 -f aiperf'."
