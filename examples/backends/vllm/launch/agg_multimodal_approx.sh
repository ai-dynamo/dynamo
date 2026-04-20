#!/bin/bash
# Native approximate MM-aware routing: Rust frontend does the URL hashing
# directly via DYN_ROUTER_MM_APPROX=1. No Python middle worker.
#
# Architecture:
#   HTTP client -> Rust Frontend (mm_routing_info + KvRouter, use_kv_events=false)
#                  -> vLLM backend

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SINGLE_GPU="${SINGLE_GPU:-true}"
NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
VLLM_SYSTEM_PORT_BASE="${VLLM_SYSTEM_PORT_BASE:-18081}"

echo "=== Native MM Approx Routing Launch ==="
echo "MODEL=${MODEL}, NUM_WORKERS=${NUM_WORKERS}, BLOCK_SIZE=${BLOCK_SIZE}"

trap 'echo; kill 0' EXIT INT TERM

wait_ready() {
    local url="$1" name="$2" timeout_s="${3:-900}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for ${name} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
            echo "${name} is ready"
            return 0
        fi
        sleep 1
    done
    return 1
}

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

for i in $(seq 1 "${NUM_WORKERS}"); do
    WORKER_PORT=$((VLLM_SYSTEM_PORT_BASE + (i - 1) * 2))
    if [[ "${SINGLE_GPU}" == "true" ]]; then GPU_ID=0; else GPU_ID=$((i - 1)); fi

    env "${COMMON_ENV[@]}" \
        "DYN_SYSTEM_PORT=${WORKER_PORT}" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    python -m dynamo.vllm \
        --model "${MODEL}" \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" &
    wait_ready "http://127.0.0.1:${WORKER_PORT}/health" "vLLM backend $i"
done

echo "=== Starting frontend (KV router, approximate mode, MM approx) ==="
env "${COMMON_ENV[@]}" \
    "DYN_ROUTER_MM_APPROX=1" \
    "DYN_ROUTER_MM_APPROX_BLOCK_SIZE=${BLOCK_SIZE}" \
    "DYN_LOG=debug" \
python -m dynamo.frontend \
    --http-port "${HTTP_PORT}" \
    --router-mode kv \
    --kv-cache-block-size "${BLOCK_SIZE}" \
    --no-router-kv-events &

sleep 5
echo "=== Native MM approx ready ==="
wait
