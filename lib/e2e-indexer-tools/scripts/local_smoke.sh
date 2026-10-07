#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EXPERIMENT ONLY: loopback plumbing smoke for the phantom-publisher topology. Never report
# numbers from this script. It proves that:
#
#   (a) publishers started BEFORE the indexer hold at the subscription gate, then deliver the
#       whole warm-up once the indexer subscribes;
#   (b) removes flow end to end and are counted (use streams with eviction);
#   (c) the indexer's delivered counters equal the publisher's sent totals (delivery_check
#       --require-exact), and driver lookups hit the phantoms' blocks.
#
#   WORK=<scratch dir> STREAMS=<phantom-stream dir> lib/e2e-indexer-tools/scripts/local_smoke.sh
#
# Starts (all on 127.0.0.1, file discovery, direct-ZMQ events, TCP request plane): one live
# mocker, one phantom publisher, one serving indexer (dynamo.router --serve-indexer) with the
# static-source patch, one frontend using the remote indexer, and one query driver. Every
# process it starts is stopped by exact PID on exit.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
WORK=${WORK:?set WORK to a scratch directory}
STREAMS=${STREAMS:?set STREAMS to a phantom-stream directory}
PY=${PY:-$ROOT/.venv/bin/python}
BIN=${BIN:-$ROOT/target/release}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BLOCK_SIZE=${BLOCK_SIZE:-16}
PHANTOMS=${PHANTOMS:-4}
PUB_PORT=${PUB_PORT:-27100}
HTTP_PORT=${HTTP_PORT:-18080}
SPEEDUP=${SPEEDUP:-1}
DURATION_S=${DURATION_S:-30}
WARMUP_BPS=${WARMUP_BPS:-2000000}
# Seconds from the publisher launch to the timed start; covers the gate hold, the indexer and
# frontend startup, and the warm-up.
LEAD_S=${LEAD_S:-90}
# Seconds the publisher must hold at the gate before the indexer starts.
GATE_HOLD_S=${GATE_HOLD_S:-10}
REPORT_S=${REPORT_S:-2}
# Streams without steady-state eviction need PLAN_ARGS=--allow-low-eviction.
read -r -a PLAN_EXTRA <<<"${PLAN_ARGS:-}"
# Page size 1 needs the SGLang mocker engine: MOCKER_ARGS="--engine-type sglang".
read -r -a MOCKER_EXTRA <<<"${MOCKER_ARGS:-}"

mkdir -p "$WORK/logs" "$WORK/kv"
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=$WORK/kv
export DYN_EVENT_PLANE=zmq
export DYN_REQUEST_PLANE=tcp
export DYN_LOG=${DYN_LOG:-info}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

start() {
  local name=$1; shift
  "$@" >"$WORK/logs/$name.log" 2>&1 &
  PIDS+=($!)
  echo "$name pid $!" | tee -a "$WORK/pids.txt"
}

wait_for() {
  local what=$1 file=$2 pattern=$3 limit=${4:-180}
  for _ in $(seq "$limit"); do
    grep -q "$pattern" "$file" 2>/dev/null && return 0
    sleep 1
  done
  echo "timed out waiting for $what ($pattern in $file)" >&2
  return 1
}

json_get() {
  "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): v=v[k]
print(v)' "$1" "$2"
}

"$BIN/phantom_plan" --streams "$STREAMS" --total-phantoms "$PHANTOMS" --speedup "$SPEEDUP" \
  --duration-s "$DURATION_S" --live-sources 1 --publishers "127.0.0.1:$PUB_PORT:$PHANTOMS" \
  --sources-out "$WORK/sources.txt" --indexer-env-out "$WORK/indexer.env" "${PLAN_EXTRA[@]}" \
  >"$WORK/plan.json"
cat "$WORK/indexer.env"

start mocker "$PY" -m dynamo.mocker --model-path "$MODEL" --model-name smoke-model \
  --block-size "$BLOCK_SIZE" --num-workers 1 "${MOCKER_EXTRA[@]}"
wait_for mocker "$WORK/logs/mocker.log" "generate" 240

# (a) The publisher starts first and must hold at the subscription gate.
START_AT=$(( $(date +%s%3N) + LEAD_S * 1000 ))
echo "$START_AT" >"$WORK/start_at_unix_ms"
start publisher "$BIN/phantom_publisher" --streams "$STREAMS" --total-phantoms "$PHANTOMS" \
  --count "$PHANTOMS" --advertise-host 127.0.0.1 --base-port "$PUB_PORT" --bind-host 127.0.0.1 \
  --speedup "$SPEEDUP" --start-at-unix-ms "$START_AT" --duration-s "$DURATION_S" \
  --warmup-blocks-per-sec "$WARMUP_BPS" --report-interval-s 2 "${PLAN_EXTRA[@]}" \
  --summary-out "$WORK/publisher-summary.json"
wait_for publisher "$WORK/logs/publisher.log" '"kind":"plan"' 120
sleep "$GATE_HOLD_S"
if grep -q '"kind":"subscribed"' "$WORK/logs/publisher.log"; then
  echo "FAIL: the publisher passed the gate before the indexer existed" >&2
  exit 1
fi
grep -q "\"pending\":$PHANTOMS" "$WORK/logs/publisher.log"
echo "gate held for ${GATE_HOLD_S}s with $PHANTOMS phantoms pending"

start indexer bash -c 'set -a; source "$1"; shift; exec "$@"' _ "$WORK/indexer.env" \
  env DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT="$WORK/accounting.json" \
  DYN_EXPERIMENT_STATIC_KV_REPORT_S="$REPORT_S" \
  DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS="$START_AT" \
  "$PY" -m dynamo.router --endpoint dynamo.backend.generate --serve-indexer \
  --router-block-size "$BLOCK_SIZE"
wait_for "publisher gate" "$WORK/logs/publisher.log" '"kind":"subscribed"' 240
echo "publisher passed the gate after the indexer subscribed"

start frontend "$PY" -m dynamo.frontend --http-port "$HTTP_PORT" --router-mode kv \
  --use-remote-indexer --kv-cache-block-size "$BLOCK_SIZE"
for _ in $(seq 240); do
  curl -sf "http://127.0.0.1:$HTTP_PORT/v1/models" | grep -q smoke-model && break
  sleep 1
done
start driver "$BIN/query_driver" --streams "$STREAMS" --total-phantoms "$PHANTOMS" \
  --speedup "$SPEEDUP" --start-at-unix-ms "$START_AT" --duration-s "$DURATION_S" \
  --component dynamo.backend --model-name smoke-model --report-interval-s 5 \
  --summary-out "$WORK/driver-summary.json"

# One real request through the frontend and the live mocker.
curl -sf "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","messages":[{"role":"user","content":"hello phantom smoke"}],"max_tokens":16}' \
  >"$WORK/frontend-response.json"
echo "frontend request ok"

wait_for driver "$WORK/driver-summary.json" "summary" $(( LEAD_S + DURATION_S + 180 ))
wait_for publisher "$WORK/publisher-summary.json" "summary" 120
PUBLISHER_PID=$(awk '/^publisher pid/ {print $3}' "$WORK/pids.txt" | tail -1)
while kill -0 "$PUBLISHER_PID" 2>/dev/null; do sleep 1; done
FINISHED=$(json_get "$WORK/publisher-summary.json" finished_unix_ms)
for _ in $(seq 60); do
  [ -f "$WORK/accounting.json" ] &&
    [ "$(json_get "$WORK/accounting.json" t_unix_ms)" -gt $(( FINISHED + 2 * REPORT_S * 1000 )) ] &&
    break
  sleep 1
done
cp "$WORK/accounting.json" "$WORK/accounting-final.json"

# (c) delivered == sent, plus the rule; (b) removes were delivered and counted.
"$BIN/delivery_check" --publisher-summary "$WORK/publisher-summary.json" \
  --indexer-accounting "$WORK/accounting-final.json" --require-exact --label smoke \
  | tee "$WORK/delivery.json"
REMOVED=$(json_get "$WORK/accounting-final.json" accounting.totals.removed_blocks)
echo "delivered removed blocks: $REMOVED"
if [ "$REMOVED" -le 0 ] && [ -z "${PLAN_ARGS:-}" ]; then
  echo "FAIL: no removes reached the indexer" >&2
  exit 1
fi
echo "publisher: $(grep -c '"kind":"interval"' "$WORK/logs/publisher.log") intervals"
cat "$WORK/driver-summary.json"
