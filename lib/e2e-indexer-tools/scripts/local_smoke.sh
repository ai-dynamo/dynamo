#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EXPERIMENT ONLY: loopback plumbing smoke for the phantom-publisher topology. Proves that
# phantom events reach the serving indexer and that driver lookups hit them. Never report
# numbers from this script.
#
#   WORK=<scratch dir> STREAMS=<phantom-stream dir> lib/e2e-indexer-tools/scripts/local_smoke.sh
#
# Starts (all on 127.0.0.1, file discovery, direct-ZMQ events, TCP request plane): one live
# mocker, one serving indexer (dynamo.router --serve-indexer) with the static-source patch,
# one frontend using the remote indexer, one phantom publisher, and one query driver. Every
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

"$BIN/phantom_plan" --streams "$STREAMS" --total-phantoms "$PHANTOMS" --speedup "$SPEEDUP" \
  --publishers "127.0.0.1:$PUB_PORT:$PHANTOMS" --sources-out "$WORK/sources.txt" \
  >"$WORK/plan.json"
cat "$WORK/sources.txt"

start mocker "$PY" -m dynamo.mocker --model-path "$MODEL" --model-name smoke-model \
  --block-size "$BLOCK_SIZE" --num-workers 1 "${MOCKER_EXTRA[@]}"
wait_for mocker "$WORK/logs/mocker.log" "generate" 240

start indexer env DYN_EXPERIMENT_STATIC_KV_SOURCES="@$WORK/sources.txt" \
  "$PY" -m dynamo.router --endpoint dynamo.backend.generate --serve-indexer \
  --router-block-size "$BLOCK_SIZE"
wait_for indexer "$WORK/logs/indexer.log" "static direct-ZMQ KV sources" 240

start frontend "$PY" -m dynamo.frontend --http-port "$HTTP_PORT" --router-mode kv \
  --use-remote-indexer --kv-cache-block-size "$BLOCK_SIZE"
for _ in $(seq 240); do
  curl -sf "http://127.0.0.1:$HTTP_PORT/v1/models" | grep -q smoke-model && break
  sleep 1
done

START_AT=$(( $(date +%s%3N) + 25000 ))
start publisher "$BIN/phantom_publisher" --streams "$STREAMS" --total-phantoms "$PHANTOMS" \
  --count "$PHANTOMS" --advertise-host 127.0.0.1 --base-port "$PUB_PORT" --bind-host 127.0.0.1 \
  --speedup "$SPEEDUP" --start-at-unix-ms "$START_AT" --duration-s "$DURATION_S" \
  --warmup-blocks-per-sec "$WARMUP_BPS" --warmup-delay-s 5 --report-interval-s 5 \
  --summary-out "$WORK/publisher-summary.json"
start driver "$BIN/query_driver" --streams "$STREAMS" --total-phantoms "$PHANTOMS" \
  --speedup "$SPEEDUP" --start-at-unix-ms "$START_AT" --duration-s "$DURATION_S" \
  --component dynamo.backend --model-name smoke-model --report-interval-s 5 \
  --summary-out "$WORK/driver-summary.json"

# One real request through the frontend and the live mocker.
curl -sf "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"smoke-model","messages":[{"role":"user","content":"hello phantom smoke"}],"max_tokens":16}' \
  >"$WORK/frontend-response.json"
echo "frontend request ok"

wait_for driver "$WORK/driver-summary.json" "summary" $(( DURATION_S + 120 ))
wait_for publisher "$WORK/publisher-summary.json" "summary" 120
echo "publisher: $(grep -c '"kind":"interval"' "$WORK/logs/publisher.log") intervals"
cat "$WORK/driver-summary.json"
