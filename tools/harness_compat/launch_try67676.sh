#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-worker, loopback-only lifecycle for the live harness compatibility lab.
# It owns only process groups it starts and writes all evidence below RUN_ROOT.

set -Eeuo pipefail

RUN_ROOT=${1:?usage: $0 RUN_ROOT}
DYNAMO_ROOT=${DYNAMO_ROOT:-/data/dynamo-wt/harness-compat-lab}
SGLANG_ROOT=${SGLANG_ROOT:-/data/sglang-wt/harness-compat-lab}
VENV=${VENV:-$DYNAMO_ROOT/.venv}
MODEL_PATH=${MODEL_PATH:-/home/nvidia/hf_cache/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b}
MODEL_NAME=${MODEL_NAME:-MiniMaxAI/MiniMax-M2}
MODEL_ALIASES=${MODEL_ALIASES:-}
GPU_SET=${GPU_SET:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}

SERVED_MODEL_NAME=$MODEL_NAME
if [[ -n $MODEL_ALIASES ]]; then
  SERVED_MODEL_NAME+=",$MODEL_ALIASES"
fi

choose_port() {
  "$VENV/bin/python" - <<'PY'
import random
import socket
for _ in range(100):
    with socket.socket() as sock:
        port = random.randrange(20_000, 30_000)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("could not reserve a dynamic loopback port below 32768")
PY
}

HTTP_PORT=${DYN_HTTP_PORT:-$(choose_port)}
SYSTEM_PORT=${DYN_SYSTEM_PORT:-$(choose_port)}
WORKER_PORT=${DYN_WORKER_PORT:-$(choose_port)}
ETCD_PORT=${DYN_ETCD_PORT:-$(choose_port)}
ETCD_NAME="harness-compat-${RUN_ROOT##*/}-etcd"
mkdir -p "$RUN_ROOT"
umask 077

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$SGLANG_ROOT/python:$DYNAMO_ROOT/components/src"
export HF_HOME=/home/nvidia/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export DYNAMO_API_KEY=compat-lab-placeholder
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_RECORDS=request_end,tool
export DYN_REQUEST_TRACE_SINKS=jsonl
export DYN_REQUEST_TRACE_FILE_FORMAT=jsonl
export DYN_REQUEST_TRACE_OUTPUT_PATH="$RUN_ROOT/request-trace.jsonl"
export DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS=100
export ETCD_ENDPOINTS="http://127.0.0.1:$ETCD_PORT"

PIDS=()
NAMES=()

start_group() {
  local name=$1 log=$2
  shift 2
  setsid "$@" >"$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
  printf '%s\n' "$!" >"$RUN_ROOT/$name.pid"
}

check_groups() {
  local index
  for ((index = 0; index < ${#PIDS[@]}; index++)); do
    kill -0 "${PIDS[$index]}" 2>/dev/null || {
      echo "${NAMES[$index]} exited; see $RUN_ROOT/${NAMES[$index]}.log" >&2
      return 1
    }
  done
}

wait_for() {
  local label=$1 timeout_s=$2 command=$3 started
  started=$(date +%s)
  until bash -c "$command" >/dev/null 2>&1; do
    check_groups
    if (( $(date +%s) - started >= timeout_s )); then
      echo "timed out waiting for $label" >&2
      return 1
    fi
    sleep 2
  done
  echo "[$(date -u +%FT%TZ)] ready: $label"
}

cleanup() {
  local index
  trap - EXIT INT TERM
  set +e
  for ((index = ${#PIDS[@]} - 1; index >= 0; index--)); do
    kill -TERM -- "-${PIDS[$index]}" 2>/dev/null || true
  done
  for ((index = ${#PIDS[@]} - 1; index >= 0; index--)); do
    for _ in $(seq 1 20); do
      kill -0 "${PIDS[$index]}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${PIDS[$index]}" 2>/dev/null || true
    wait "${PIDS[$index]}" 2>/dev/null || true
  done
  docker rm -f "$ETCD_NAME" >/dev/null 2>&1 || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >"$RUN_ROOT/gpu-after-cleanup.csv" 2>&1 || true
  date -u +%FT%TZ >"$RUN_ROOT/stopped-at"
}
trap cleanup EXIT
trap 'cleanup; exit 0' INT TERM

git -C "$DYNAMO_ROOT" rev-parse HEAD >"$RUN_ROOT/dynamo-sha"
git -C "$SGLANG_ROOT" rev-parse HEAD >"$RUN_ROOT/sglang-sha"
git -C "$DYNAMO_ROOT" diff --binary >"$RUN_ROOT/dynamo.diff"
git -C "$SGLANG_ROOT" diff --binary >"$RUN_ROOT/sglang.diff"
date -u +%FT%TZ >"$RUN_ROOT/started-at"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >"$RUN_ROOT/gpu-before.csv"
printf '{"http_port":%s,"system_port":%s,"worker_port":%s,"etcd_port":%s,"model":"%s"}\n' "$HTTP_PORT" "$SYSTEM_PORT" "$WORKER_PORT" "$ETCD_PORT" "$MODEL_NAME" >"$RUN_ROOT/endpoint.json"
printf '%s\n' "$MODEL_ALIASES" >"$RUN_ROOT/model-aliases.txt"

docker rm -f "$ETCD_NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$ETCD_NAME" -p "127.0.0.1:$ETCD_PORT:2379" -e ALLOW_NONE_AUTHENTICATION=yes bitnamilegacy/etcd:3.6.1 >"$RUN_ROOT/etcd.container-id"
wait_for etcd 60 "curl --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:$ETCD_PORT/health"

start_group frontend "$RUN_ROOT/frontend.log" env DYN_HTTP_PORT="$HTTP_PORT" \
  "$VENV/bin/python" -m dynamo.frontend \
  --http-port "$HTTP_PORT" \
  --router-mode round-robin \
  --shared-cache-type none \
  --enable-anthropic-api \
  --strip-anthropic-preamble \
  --enable-streaming-tool-dispatch
wait_for frontend 120 "curl --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:$HTTP_PORT/health"

start_group worker "$RUN_ROOT/worker.log" env CUDA_VISIBLE_DEVICES="$GPU_SET" DYN_SYSTEM_PORT="$SYSTEM_PORT" \
  "$VENV/bin/python" -m dynamo.sglang \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tp-size "$TP_SIZE" \
  --port "$WORKER_PORT" \
  --page-size 16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.72 \
  --attention-backend fa3 \
  --decode-attention-backend flashinfer \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled \
  --skip-tokenizer-init \
  --enable-metrics \
  --trust-remote-code \
  --dyn-tool-call-parser minimax_m2 \
  --dyn-reasoning-parser minimax_append_think

wait_for model-registration 1800 "curl --connect-timeout 2 --max-time 10 -fsS http://127.0.0.1:$HTTP_PORT/v1/models | jq -e --arg model '$MODEL_NAME' '.data | any(.id == \$model)'"
for model_alias in ${MODEL_ALIASES//,/ }; do
  wait_for "model-alias:$model_alias" 1800 "curl --connect-timeout 2 --max-time 10 -fsS http://127.0.0.1:$HTTP_PORT/v1/models | jq -e --arg model '$model_alias' '.data | any(.id == \$model)'"
done
SMOKE_BODY=$(jq -nc --arg model "$MODEL_NAME" '{model:$model,messages:[{role:"user",content:"Reply with OK."}],max_tokens:1,stream:false}')
wait_for inference 300 "curl --connect-timeout 2 --max-time 120 -fsS http://127.0.0.1:$HTTP_PORT/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer compat-lab-placeholder' -d '$SMOKE_BODY'"
touch "$RUN_ROOT/READY"
echo "$RUN_ROOT"

while true; do
  check_groups
  sleep 5
done
