#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validate MM-aware KV routing in the frontend (no mm_router_worker).
#
# Architecture:
#   Frontend (--dyn-chat-processor vllm, --router-mode kv) --> vLLM backend
#
# The frontend uses vllm_processor.py which computes mm_routing_info
# directly: downloads images via vLLM's renderer, unwraps MediaWithBytes
# to get PIL images, hashes with compute_mm_uuids_from_images, and builds
# per-block mm_infos for the KvRouter.
#
# Key log lines to watch in frontend output:
#   Startup:
#     MM image_token_id=<N> block_size=16 (None=MM routing disabled)
#
#   Per MM request:
#     [mm-routing] placeholder_tokens=<A> vllm_expanded_tokens=<B> ... pil_images=1 image_token_id=<N>
#     [mm-routing] mm_hashes=[...] image_ranges=[(...)]
#     [mm-routing] built mm_routing_info: routing_tokens=<B> blocks=<C>
#
#   KV routing (from Rust, requires DYN_LOG=debug):
#     [ROUTING] Best: worker_... with X/Y blocks overlap

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

VLLM_SYSTEM_PORT="${VLLM_SYSTEM_PORT:-18081}"
VLLM_KV_EVENT_PORT="${VLLM_KV_EVENT_PORT:-20080}"
VLLM_SYSTEM_PORT2="${VLLM_SYSTEM_PORT2:-18082}"
VLLM_KV_EVENT_PORT2="${VLLM_KV_EVENT_PORT2:-20081}"

# GPU assignment: worker 0 on GPU 0, worker 1 on GPU 1 by default.
# Single-GPU: override with WORKER0_GPUS=0 WORKER1_GPUS=0 and reduce
# GPU_MEMORY_UTILIZATION to ~0.4 so both fit.
WORKER0_GPUS="${WORKER0_GPUS:-0}"
WORKER1_GPUS="${WORKER1_GPUS:-0}"

VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

# Set deterministic hash seed for KV event IDs (required for consistent block hashes)
export PYTHONHASHSEED=0

echo "=== MM KV Frontend Validation (2 workers) ==="
echo "Architecture: Frontend(vllm+kv) --> vLLM worker-0 + worker-1  [no mm_router_worker]"
echo "MODEL=${MODEL}"
echo "HTTP_PORT=${HTTP_PORT}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "VLLM_SYSTEM_PORT=${VLLM_SYSTEM_PORT}  VLLM_SYSTEM_PORT2=${VLLM_SYSTEM_PORT2}"
echo "WORKER0_GPUS=${WORKER0_GPUS}  WORKER1_GPUS=${WORKER1_GPUS}"
echo

PIDS=()

cleanup() {
    echo
    echo "Cleaning up..."
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_ready() {
    local url="$1"
    local name="$2"
    local timeout_s="${3:-300}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for ${name} at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
            echo "${name} is ready"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Timed out waiting for ${name}" >&2
    return 1
}

wait_frontend() {
    local url="$1"
    local timeout_s="${2:-120}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for frontend at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            echo "Frontend is ready"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Timed out waiting for frontend" >&2
    return 1
}

echo "Prerequisite: etcd + NATS must be running."
echo "  docker compose -f deploy/docker-compose.yml up -d"
echo

COMMON_ENV=(
    "DYN_NAMESPACE=dynamo"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

# ---------------------------------------------------------------------------
# 1a. vLLM worker-0
# ---------------------------------------------------------------------------
echo "=== Starting vLLM worker-0 (GPU ${WORKER0_GPUS}, port ${VLLM_SYSTEM_PORT}) ==="
CUDA_VISIBLE_DEVICES="${WORKER0_GPUS}" \
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${VLLM_SYSTEM_PORT}" \
    "DYN_VLLM_KV_EVENT_PORT=${VLLM_KV_EVENT_PORT}" \
    python -m dynamo.vllm \
        --model "${MODEL}" \
        --enable-prefix-caching \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        ${VLLM_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${VLLM_SYSTEM_PORT}/health" "vLLM worker-0" 900

# ---------------------------------------------------------------------------
# 1b. vLLM worker-1
# ---------------------------------------------------------------------------
echo
echo "=== Starting vLLM worker-1 (GPU ${WORKER1_GPUS}, port ${VLLM_SYSTEM_PORT2}) ==="
CUDA_VISIBLE_DEVICES="${WORKER1_GPUS}" \
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${VLLM_SYSTEM_PORT2}" \
    "DYN_VLLM_KV_EVENT_PORT=${VLLM_KV_EVENT_PORT2}" \
    python -m dynamo.vllm \
        --model "${MODEL}" \
        --enable-prefix-caching \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        ${VLLM_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${VLLM_SYSTEM_PORT2}/health" "vLLM worker-1" 900

# ---------------------------------------------------------------------------
# 2. Frontend: vllm chat processor + kv router, direct to backend
#    DYN_LOG=debug enables [ROUTING] overlap logs from Rust kv_router
# ---------------------------------------------------------------------------
echo
echo "=== Starting frontend (--chat-processor vllm --router-mode kv) ==="
env "${COMMON_ENV[@]}" \
    "DYN_LOG=debug" \
    python -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode kv \
        --kv-cache-block-size "${BLOCK_SIZE}" \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend "http://127.0.0.1:${HTTP_PORT}/v1/models" 120

echo
echo "=== All services ready ==="
echo "Frontend: http://127.0.0.1:${HTTP_PORT}"
echo
echo "Send the same request twice and compare frontend logs:"
echo "  1st request: [mm-routing] built mm_routing_info: routing_tokens=<N> blocks=<M>"
echo "  2nd request: [ROUTING] Best: worker_... with X/Y blocks overlap  (X>0 = cache hit)"
echo
echo "Test command (run twice):"
cat <<EOF
curl http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image briefly."},
        {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/test2017/000000000001.jpg"}}
      ]
    }],
    "max_tokens": 50
  }'
EOF
echo
echo "Press Ctrl+C to stop."
wait
