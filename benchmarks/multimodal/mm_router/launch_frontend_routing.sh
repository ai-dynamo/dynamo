#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for NEW Frontend MM Routing benchmark:
#   Frontend (vLLM processor + KvRouter + ImageLoader + NIXL) -> vLLM backend workers
#
# No separate MM Router Worker. The frontend handles:
#   1. Image downloading via ImageLoader (LRU cache + in-flight dedup)
#   2. HF processor via vLLM's process_inputs() (model-agnostic)
#   3. MM routing via mm_features -> block_mm_infos -> KvRouter
#   4. NIXL transfer of mm_kwargs to skip backend HF processor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-/workspace}"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Defaults for the 30B FP8 model. Each worker is pinned to one GPU (CUDA_VISIBLE_DEVICES=i).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# Set SINGLE_GPU=1 to pin all workers to GPU 0 (single-GPU benchmarking).
# Default (0): each worker gets its own GPU (worker i -> GPU i-1).
SINGLE_GPU="${SINGLE_GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-100426}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE:-500}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
NUM_FRONTENDS="${NUM_FRONTENDS:-1}"

VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

echo "=== Frontend MM Routing Benchmark (${NUM_WORKERS} Workers) ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "MODEL=${MODEL}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "SINGLE_GPU=${SINGLE_GPU}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "DYN_MM_IMAGE_CACHE_SIZE=${DYN_MM_IMAGE_CACHE_SIZE}"
echo "PREPROCESS_WORKERS=${PREPROCESS_WORKERS}"
echo "NUM_FRONTENDS=${NUM_FRONTENDS}"
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
            echo "${name} is ready"; return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for ${name} (${url})" >&2; return 1
}

wait_frontend_models() {
    local url="$1"
    local timeout_s="${2:-240}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for frontend at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            echo "Frontend is ready"; return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for frontend (${url})" >&2; return 1
}

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

# How many workers to launch before waiting for the gate.
# Default 2: start 2 workers, wait for the first to load weights, then continue.
WORKER_LAUNCH_PARALLELISM="${WORKER_LAUNCH_PARALLELISM:-2}"
# Gate type: "ready" (full health check), "weights_loaded" (faster, overlap init)
WORKER_LAUNCH_GATE="${WORKER_LAUNCH_GATE:-weights_loaded}"

BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/dynamo_backend_logs}"
mkdir -p "${BACKEND_LOG_DIR}"

wait_log_regex() {
    local log_file="$1"
    local pattern="$2"
    local label="$3"
    local timeout_s="${4:-900}"
    local pid="${5:-}"
    local deadline=$((SECONDS + timeout_s))

    echo "Waiting for ${label} (pattern: ${pattern}) ..."
    while (( SECONDS < deadline )); do
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "Process ${pid} died while waiting for ${label}" >&2
            return 1
        fi
        if [[ -f "${log_file}" ]] && grep -qE "${pattern}" "${log_file}" 2>/dev/null; then
            echo "${label} - gate reached"
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for ${label}" >&2
    return 1
}

declare -a BACKEND_PIDS=()
declare -a BACKEND_PORTS=()
declare -a BACKEND_LOGS=()

# ---------------------------------------------------------------------------
# Start backend workers (parallel loading with gating)
# ---------------------------------------------------------------------------
for i in $(seq 1 "${NUM_WORKERS}"); do
    if [[ "${SINGLE_GPU}" == "1" ]]; then
        gpu_id=0
    else
        gpu_id=$((i - 1))
    fi
    system_port=$((18079 + i * 2))
    kv_port=$((20079 + i))
    log_file="${BACKEND_LOG_DIR}/worker_${i}.log"

    echo "=== Starting vLLM backend worker #${i} (GPU ${gpu_id}, port ${system_port}) ==="
    env "${COMMON_ENV[@]}" \
        "CUDA_VISIBLE_DEVICES=${gpu_id}" \
        "DYN_SYSTEM_PORT=${system_port}" \
        "${PYTHON_BIN}" -m dynamo.vllm \
            --model "${MODEL}" \
            --enable-prefix-caching \
            --enable-multimodal \
            --block-size "${BLOCK_SIZE}" \
            --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${kv_port}\",\"enable_kv_cache_events\":true}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            ${VLLM_EXTRA_ARGS} > "${log_file}" 2>&1 &
    PIDS+=($!)
    BACKEND_PIDS+=($!)
    BACKEND_PORTS+=("${system_port}")
    BACKEND_LOGS+=("${log_file}")

    # Gate: wait for earlier workers to pass the gate before starting more
    idx=$((i - 1))
    if (( i >= WORKER_LAUNCH_PARALLELISM )); then
        gate_idx=$(( i - WORKER_LAUNCH_PARALLELISM ))
        case "${WORKER_LAUNCH_GATE}" in
            ready)
                wait_ready "http://127.0.0.1:${BACKEND_PORTS[$gate_idx]}/health" "vLLM backend #$((gate_idx + 1))" 900
                ;;
            weights_loaded)
                wait_log_regex \
                    "${BACKEND_LOGS[$gate_idx]}" \
                    "Loading weights took|Model loading took" \
                    "vLLM backend #$((gate_idx + 1)) weights_loaded" \
                    900 \
                    "${BACKEND_PIDS[$gate_idx]}"
                ;;
        esac
    fi
    echo
done

# Wait for ALL workers to be fully ready
echo "=== Waiting for all ${NUM_WORKERS} workers to be ready ==="
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    wait_ready "http://127.0.0.1:${BACKEND_PORTS[$i]}/health" "vLLM backend #$((i + 1))" 900
done
echo "All backend workers are ready."

# ---------------------------------------------------------------------------
# Start frontend(s) with vLLM processor + KV router
# ---------------------------------------------------------------------------
FRONTEND_POOL_ARGS=""
if [[ "${PREPROCESS_WORKERS}" -gt 0 ]]; then
    FRONTEND_POOL_ARGS="--dyn-preprocess-workers ${PREPROCESS_WORKERS}"
fi

FRONTEND_SYSTEM_PORT_BASE="${FRONTEND_SYSTEM_PORT_BASE:-9080}"

for f in $(seq 1 "${NUM_FRONTENDS}"); do
    FE_HTTP_PORT=$((HTTP_PORT + f - 1))
    FE_SYSTEM_PORT=$((FRONTEND_SYSTEM_PORT_BASE + f - 1))

    RESET_ARGS=""
    if [[ "$f" -eq 1 ]]; then
        RESET_ARGS="--router-reset-states"
    fi

    SYNC_ARGS=""
    if [[ "${NUM_FRONTENDS}" -gt 1 ]]; then
        SYNC_ARGS="--router-replica-sync"
    fi

    echo "=== Starting frontend replica ${f} (HTTP ${FE_HTTP_PORT}, system ${FE_SYSTEM_PORT}) ==="
    env "${COMMON_ENV[@]}" \
        "DYN_LOG=info" \
        "DYN_SYSTEM_PORT=${FE_SYSTEM_PORT}" \
        "${PYTHON_BIN}" -m dynamo.frontend \
            --http-port "${FE_HTTP_PORT}" \
            --dyn-chat-processor vllm \
            --router-mode kv \
            --kv-cache-block-size "${BLOCK_SIZE}" \
            --model-name "${MODEL}" \
            ${FRONTEND_POOL_ARGS} \
            ${RESET_ARGS} \
            ${SYNC_ARGS} \
            ${FRONTEND_EXTRA_ARGS} &
    PIDS+=($!)
    wait_frontend_models "http://127.0.0.1:${FE_HTTP_PORT}/v1/models" 300
done

# Wait for first frontend processor to warm up
echo "Warming up frontend processor..."
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

# Build URL list for aiperf (space-separated for multiple --url flags)
FRONTEND_URLS=""
for f in $(seq 1 "${NUM_FRONTENDS}"); do
    FE_HTTP_PORT=$((HTTP_PORT + f - 1))
    FRONTEND_URLS="${FRONTEND_URLS} http://127.0.0.1:${FE_HTTP_PORT}"
done
FRONTEND_URLS="${FRONTEND_URLS# }"  # trim leading space
# Export so run_sweep.sh can pick it up
export AIPERF_URL="${FRONTEND_URLS}"

echo
echo "=== All services are ready ==="
for f in $(seq 1 "${NUM_FRONTENDS}"); do
    echo "Frontend ${f}: http://127.0.0.1:$((HTTP_PORT + f - 1))"
done
for i in $(seq 1 "${NUM_WORKERS}"); do
    echo "Worker ${i}: http://127.0.0.1:$((18079 + i * 2))/health"
done
echo
echo "Architecture: ${NUM_FRONTENDS}x Frontend (vLLM processor + KvRouter + NIXL) -> ${NUM_WORKERS}x vLLM backend"
echo "AIPERF_URL=${AIPERF_URL}"
echo
echo "Press Ctrl+C to stop all services"

wait
