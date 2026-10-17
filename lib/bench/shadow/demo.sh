#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shadow tap demo: a frontend with two taps, mocker workers, AIPerf load, and
# one consumer per tap. It needs no etcd and no NATS: discovery is a directory
# and the event plane is ZMQ.
#
# Run from the repository root inside a venv that has `ai-dynamo` and `aiperf`.
#
#   lib/bench/shadow/demo.sh [request-count] [concurrency] [isl] [osl]
#
# STREAMING= sends non-streaming requests. The taps publish one record per
# request in both cases.
#
# AIPERF_EXTRA adds AIPerf flags, for example a cancellation run:
#   AIPERF_EXTRA='--request-cancellation-rate 25 --request-cancellation-delay 0.05'
set -euo pipefail

REQUESTS=${1:-200}
CONCURRENCY=${2:-16}
ISL=${3:-512}
OSL=${4:-32}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
OUT=${OUT:-$(mktemp -d -t shadow-demo.XXXXXX)}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="$OUT/discovery"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq

cargo build -p dynamo-bench --no-default-features --features shadow-consumer --bin shadow_consumer
CONSUMER=$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/debug/shadow_consumer

pids=()
cleanup() {
  kill -INT "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

python3 -m dynamo.mocker --model-path "$MODEL" --num-workers 2 --speedup-ratio 10 \
  --discovery-backend file --request-plane tcp >"$OUT/mocker.log" 2>&1 &
pids+=($!)

DYN_SHADOW_TAP_CONFIG="$HERE/taps.yaml" python3 -m dynamo.frontend --http-port "$PORT" \
  --discovery-backend file --request-plane tcp >"$OUT/frontend.log" 2>&1 &
pids+=($!)

for topic in kv-history-tracker shadow-replay; do
  "$CONSUMER" --topic "$topic" --summary-path "$OUT/$topic.json" >"$OUT/$topic.log" 2>&1 &
  pids+=($!)
done

echo "waiting for the model to register (logs in $OUT)"
for _ in $(seq 1 120); do
  if curl -sf "localhost:$PORT/v1/models" | grep -q "$MODEL"; then break; fi
  sleep 1
done
curl -sf "localhost:$PORT/v1/models" | grep -q "$MODEL" || { echo "model never registered"; exit 1; }
# Subscribers find the tap publishers through discovery; give them a moment.
sleep 3

aiperf profile --model "$MODEL" --url "localhost:$PORT" --endpoint-type chat ${STREAMING---streaming} \
  --request-count "$REQUESTS" --concurrency "$CONCURRENCY" \
  --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OSL" --extra-inputs ignore_eos:true \
  --artifact-dir "$OUT/aiperf" ${AIPERF_EXTRA:-} >"$OUT/aiperf.log" 2>&1 || echo "aiperf exited non-zero; see $OUT/aiperf.log"

sleep 2
echo "== frontend metrics"
curl -s "localhost:$PORT/metrics" | grep -E '^dynamo_frontend_(input|output)_sequence_tokens_(sum|count)|^dynamo_frontend_requests_total' || true
for topic in kv-history-tracker shadow-replay; do
  echo "== $topic"
  cat "$OUT/$topic.json"
  echo
done
