#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for vLLM MM Frontend Routing demo:
#   Frontend (vllm processor + KvRouter + ImageLoader) -> vLLM backend
#
# This replaces the separate MM Router Worker approach.  The frontend runs
# vLLM's HF processor for image token expansion and hash computation, builds
# mm_routing_info, and uses KvRouter directly for MM-aware routing.
#
# Benefits over mm_router_worker:
#   - No extra network hop for base64 image data
#   - Model-agnostic (no Qwen-specific token expansion)
#   - Single HF processor call (frontend only; backend skips via NIXL, when enabled)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"            # Must match vLLM backend KV block size
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"  # Split GPU between 2 workers
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"   # Reduced for 2 workers on 1 GPU
NUM_WORKERS="${NUM_WORKERS:-2}"          # Number of backend workers
# --single-gpu / SINGLE_GPU: Packs all workers onto GPU 0 for functional
# testing on machines with a single GPU.  Reduces performance by sharing
# GPU memory between workers.
SINGLE_GPU="${SINGLE_GPU:-false}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"  # Frontend preprocess worker pool size (0=disabled)
NUM_FRONTENDS="${NUM_FRONTENDS:-1}"           # Number of frontend replicas (>1 for parallel HF processing)

# KV cache override for parallel-safe GPU memory control
KV_BYTES="${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}"
if [[ -n "$KV_BYTES" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.01"
else
    GPU_MEM_ARGS="--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
fi

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

VLLM_SYSTEM_PORT_BASE="${VLLM_SYSTEM_PORT_BASE:-18081}"

# ImageLoader cache size (number of images kept in-memory LRU)
export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE:-32}"

# Extra args (word-splitting is intentional for shell-style overrides)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

echo "=== vLLM MM Frontend Routing Launch Script ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "MODEL=${MODEL}"
echo "NAMESPACE=${NAMESPACE}"
echo "HTTP_PORT=${HTTP_PORT}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "SINGLE_GPU=${SINGLE_GPU}"
echo "PREPROCESS_WORKERS=${PREPROCESS_WORKERS}"
echo "NUM_FRONTENDS=${NUM_FRONTENDS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "DYN_MM_IMAGE_CACHE_SIZE=${DYN_MM_IMAGE_CACHE_SIZE}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "VLLM_SYSTEM_PORT_BASE=${VLLM_SYSTEM_PORT_BASE}"
echo

PIDS=()

cleanup() {
    echo
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_ready() {
    local url="$1"
    local name="$2"
    local timeout_s="${3:-240}"
    local deadline=$((SECONDS + timeout_s))

    echo "Waiting for ${name} at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
            echo "${name} is ready"
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for ${name} (${url})" >&2
    return 1
}

wait_frontend_models() {
    local url="$1"
    local timeout_s="${2:-240}"
    local deadline=$((SECONDS + timeout_s))

    echo "Waiting for frontend models API at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            echo "Frontend is ready"
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for frontend (${url})" >&2
    return 1
}

echo "Prerequisite: start etcd and NATS yourself before running this script."
echo "Example:"
echo "  docker compose -f deploy/docker-compose.yml up -d"
echo

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

for i in $(seq 1 "${NUM_WORKERS}"); do
    WORKER_PORT=$((18079 + i * 2))
    KV_EVENTS_PORT=$((${KV_EVENTS_PORT_BASE:-29080} + i - 1))

    if [[ "${SINGLE_GPU}" == "true" ]]; then
        GPU_ID=0
    else
        GPU_ID=$((i - 1))
    fi

    echo
    echo "=== Starting vLLM backend worker $i (GPU ${GPU_ID}, port ${WORKER_PORT}, kv_events ${KV_EVENTS_PORT}) ==="
    env "${COMMON_ENV[@]}" \
        "DYN_SYSTEM_PORT=${WORKER_PORT}" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    python -m dynamo.vllm \
            --model "${MODEL}" \
            --enable-multimodal \
            --block-size "${BLOCK_SIZE}" \
            --enforce-eager \
            --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\",\"enable_kv_cache_events\":true}" \
            $GPU_MEM_ARGS \
            --max-model-len "${MAX_MODEL_LEN}" \
            ${VLLM_EXTRA_ARGS} &
    PIDS+=($!)
    # Wait for this worker before starting the next one to avoid ZMQ port races.
    wait_ready "http://127.0.0.1:${WORKER_PORT}/health" "vLLM backend $i" 900
done

echo
echo "=== Starting frontend (with vLLM processor + KV router) ==="
# Key flags:
#   --dyn-chat-processor vllm    : run vLLM's HF processor in the frontend
#   --router-mode kv             : use KvRouter for MM-aware routing
#   --kv-cache-block-size        : must match backend's --block-size
#
# The frontend's VllmProcessor:
#   1. Pre-downloads images via ImageLoader (LRU cache + dedup)
#   2. Runs vLLM's process_inputs() → mm_features with hashes + placeholders
#   3. Builds mm_routing_info from mm_features → passes to KvRouter
#   4. Forwards mm_hashes to backend for hash consistency
FRONTEND_POOL_ARGS=""
if [[ "${PREPROCESS_WORKERS}" -gt 0 ]]; then
    FRONTEND_POOL_ARGS="--dyn-preprocess-workers ${PREPROCESS_WORKERS}"
fi

FRONTEND_SYSTEM_PORT_BASE="${FRONTEND_SYSTEM_PORT_BASE:-9080}"

for f in $(seq 1 "${NUM_FRONTENDS}"); do
    FE_HTTP_PORT=$((HTTP_PORT + f - 1))
    FE_SYSTEM_PORT=$((FRONTEND_SYSTEM_PORT_BASE + f - 1))

    # Only reset states on the first replica to avoid wiping shared state.
    RESET_ARGS=""
    if [[ "$f" -eq 1 ]]; then
        RESET_ARGS="--router-reset-states"
    fi

    # Enable replica sync when running multiple frontends.
    SYNC_ARGS=""
    if [[ "${NUM_FRONTENDS}" -gt 1 ]]; then
        SYNC_ARGS="--router-replica-sync"
    fi

    echo
    echo "=== Starting frontend replica ${f} (HTTP ${FE_HTTP_PORT}, system ${FE_SYSTEM_PORT}) ==="
    env "${COMMON_ENV[@]}" \
        "DYN_LOG=debug" \
        "DYN_SYSTEM_PORT=${FE_SYSTEM_PORT}" \
        python -m dynamo.frontend \
            --http-port "${FE_HTTP_PORT}" \
            --dyn-chat-processor vllm \
            --router-mode kv \
            --kv-cache-block-size "${BLOCK_SIZE}" \
            ${FRONTEND_POOL_ARGS} \
            ${RESET_ARGS} \
            ${SYNC_ARGS} \
            --model-name "${MODEL}" \
            ${FRONTEND_EXTRA_ARGS} &
    PIDS+=($!)
    wait_frontend_models "http://127.0.0.1:${FE_HTTP_PORT}/v1/models" 300
done

# Wait until the first frontend can serve a real request (processor loaded).
echo "Waiting for frontend processor to initialize (this may take a while for custom models)..."
DEADLINE=$((SECONDS + 300))
while (( SECONDS < DEADLINE )); do
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "Frontend processor is ready"
        break
    fi
    sleep 2
done
if (( SECONDS >= DEADLINE )); then
    echo "Warning: Frontend processor may not be fully ready (timed out)" >&2
fi

echo
echo "=== All services are ready ==="
for f in $(seq 1 "${NUM_FRONTENDS}"); do
    echo "Frontend ${f}: http://127.0.0.1:$((HTTP_PORT + f - 1))"
done
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Worker $i: http://127.0.0.1:$((VLLM_SYSTEM_PORT_BASE + i))/health"
done
echo
echo "Architecture: ${NUM_FRONTENDS}x Frontend (vLLM processor + KvRouter) -> ${NUM_WORKERS}x vLLM backend"
echo "  - No separate MM Router Worker needed"
echo "  - ImageLoader LRU cache size: ${DYN_MM_IMAGE_CACHE_SIZE}"
echo
echo "Test routing: send the same image request 3 times."
echo "  - Request 1: routed to any worker (no cache yet)"
echo "  - Request 2: routed to SAME worker (KV cache overlap from request 1)"
echo "  - Request 3 with different image: may route to the OTHER worker"
echo
echo "Watch for 'Selected worker' in logs to see routing decisions change."
echo
echo "Example (same image twice, then different image):"
echo "  # Request 1 - image A"
echo "  curl http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://images.cocodataset.org/test2017/000000000001.jpg\"}}]}],\"max_tokens\":32}'"
echo
echo "  # Request 2 - same image A (should route to same worker)"
echo "  curl http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://images.cocodataset.org/test2017/000000000001.jpg\"}}]}],\"max_tokens\":32}'"
echo
echo "  # Request 3 - different image B (may route to other worker)"
echo "  curl http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://images.cocodataset.org/test2017/000000000016.jpg\"}}]}],\"max_tokens\":32}'"
echo
echo "Press Ctrl+C to stop all services"

wait
