#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Tunables (all overridable via env). Server + both engines must agree on
# GMS_SOCKET_DIR — that is how they find the same per-GPU sockets.
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export GMS_SOCKET_DIR="${GMS_SOCKET_DIR:-/tmp/gms-demo}"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}"
export NATS_SERVER="${NATS_SERVER:-nats://localhost:4222}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
PRIMARY_SYSTEM_PORT=8081 SHADOW_SYSTEM_PORT=8082   # engine control / health
PRIMARY_NIXL_PORT=8281    SHADOW_NIXL_PORT=8282     # nixl side channel
PRIMARY_KV_PORT=8181      SHADOW_KV_PORT=8182       # kv-event publisher

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; kill -KILL -"${PRIMARY_PGID:-0}" 2>/dev/null' EXIT

# Poll an HTTP endpoint until it answers 200, or give up after ~300s.
ready() { for _ in $(seq 1 300); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

engine() { # engine <system_port> <nixl_port> <kv_port> [extra args...]
  setsid env DYN_SYSTEM_PORT="$1" VLLM_NIXL_SIDE_CHANNEL_PORT="$2" \
    python -m dynamo.vllm --model "$MODEL" --load-format gms --enforce-eager \
    --enable-sleep-mode --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$3\",\"enable_kv_cache_events\":true}" \
    "${@:4}"
}

COMPLETION="{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":20}"

mkdir -p "$GMS_SOCKET_DIR"
etcd --data-dir /tmp/gms-demo-etcd >/tmp/gms-demo-etcd.log 2>&1 & PIDS+=($!)
nats-server -js --store_dir /tmp/gms-demo-nats >/tmp/gms-demo-nats.log 2>&1 & PIDS+=($!)
python -m gpu_memory_service.cli.server >/tmp/gms-demo-gms.log 2>&1 & PIDS+=($!)
python -m dynamo.frontend --http-port "$FRONTEND_PORT" >/tmp/gms-demo-frontend.log 2>&1 & PIDS+=($!)
sleep 5

# PRIMARY is the RW writer: loads the model once and publishes weights into GMS.
engine "$PRIMARY_SYSTEM_PORT" "$PRIMARY_NIXL_PORT" "$PRIMARY_KV_PORT" \
  >/tmp/gms-demo-primary.log 2>&1 & PRIMARY_PGID=$!
# SHADOW is the RO importer: attaches to the resident weights, no second disk load.
engine "$SHADOW_SYSTEM_PORT" "$SHADOW_NIXL_PORT" "$SHADOW_KV_PORT" \
  --model-loader-extra-config '{"gms_read_only": true}' \
  >/tmp/gms-demo-shadow.log 2>&1 & PIDS+=($!)

ready "localhost:$PRIMARY_SYSTEM_PORT/health" && ready "localhost:$SHADOW_SYSTEM_PORT/health"
echo "primary + shadow ready"

# Park the shadow, then prove the primary serves through the frontend.
curl -fsS -X POST "localhost:$SHADOW_SYSTEM_PORT/engine/control/sleep" -H 'content-type: application/json' -d '{"level":2}'
curl -fsS -X POST "localhost:$FRONTEND_PORT/v1/completions" -H 'content-type: application/json' -d "$COMPLETION" >/dev/null

echo "killing primary"
kill -KILL -"$PRIMARY_PGID"

echo "waking shadow"
curl -fsS -X POST "localhost:$SHADOW_SYSTEM_PORT/engine/control/wake_up" -H 'content-type: application/json' -d '{}'

echo "verifying takeover"
for _ in $(seq 1 30); do
  body="$(curl -fsS -X POST "localhost:$FRONTEND_PORT/v1/completions" -H 'content-type: application/json' -d "$COMPLETION" 2>/dev/null)" \
    && [[ "$body" == *'"choices"'* ]] && { echo "TAKEOVER OK"; exit 0; }
  sleep 2
done
echo "TAKEOVER FAILED" >&2; exit 1
