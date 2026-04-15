#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for a local MM-router demo with vLLM backend workers:
#   Frontend (round-robin) -> MM Router Worker -> vLLM backend #1 … #NUM_WORKERS
#
# Each worker is pinned to its own GPU via CUDA_VISIBLE_DEVICES.
# This script assumes etcd + NATS are already running (see deploy/docker-compose.yml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-/workspace}"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}" # Must match vLLM backend KV block size
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

# Per-worker ports are computed as:
#   VLLM system port : 18079 + i*2  (18081, 18083, 18085, ...)
#   KV event port    : 20079 + i    (20080, 20081, 20082, ...)
#   served model name: ${MODEL}__internal_${i}
# Override any individual worker by exporting VLLMi_SYSTEM_PORT, KV_EVENT_PORT_i,
# or VLLMi_SERVED_MODEL_NAME before running this script.
MM_ROUTER_SYSTEM_PORT="${MM_ROUTER_SYSTEM_PORT:-18082}"

MM_ROUTER_COMPONENT="${MM_ROUTER_COMPONENT:-mm_router}"
BACKEND_COMPONENT="${BACKEND_COMPONENT:-backend}" # dynamo.vllm default
MM_ROUTER_REWRITE_IMAGE_URLS="${MM_ROUTER_REWRITE_IMAGE_URLS:-0}"

# Extra args (word-splitting intentional for shell-style overrides)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"
MM_ROUTER_EXTRA_ARGS="${MM_ROUTER_EXTRA_ARGS:-}"

echo "=== vLLM MM Router (${NUM_WORKERS} Workers) Launch Script ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL=${MODEL}"
echo "NAMESPACE=${NAMESPACE}"
echo "HTTP_PORT=${HTTP_PORT}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "SINGLE_GPU=${SINGLE_GPU}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
for i in $(seq 1 "${NUM_WORKERS}"); do
    sys_var="VLLM${i}_SYSTEM_PORT"; sp="${!sys_var:-$((18079 + i * 2))}"
    kv_var="KV_EVENT_PORT_${i}";    kp="${!kv_var:-$((20079 + i))}"
    echo "VLLM${i}_SYSTEM_PORT=${sp} (KV events ${kp}, GPU $((i-1)))"
done
echo "MM_ROUTER_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}"
echo "MM_ROUTER_COMPONENT=${MM_ROUTER_COMPONENT}"
echo "BACKEND_COMPONENT=${BACKEND_COMPONENT}"
echo "MM_ROUTER_REWRITE_IMAGE_URLS=${MM_ROUTER_REWRITE_IMAGE_URLS}"
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

echo "Prerequisite: start etcd and NATS first."
echo "Example:"
echo "  docker compose -f deploy/docker-compose.yml up -d"
echo

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

PYTHONPATH_VALUE="${DYNAMO_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi

WORKER_LAUNCH_PARALLELISM="${WORKER_LAUNCH_PARALLELISM:-2}"
WORKER_LAUNCH_GATE="${WORKER_LAUNCH_GATE:-weights_loaded}"
BACKEND_LOG_DIR="${BACKEND_LOG_DIR:-/tmp/dynamo_backend_logs}"
mkdir -p "${BACKEND_LOG_DIR}"

wait_log_regex() {
    local log_file="$1" pattern="$2" label="$3" timeout_s="${4:-900}" pid="${5:-}"
    local deadline=$((SECONDS + timeout_s))
    echo "Waiting for ${label} (pattern: ${pattern}) ..."
    while (( SECONDS < deadline )); do
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "Process ${pid} died while waiting for ${label}" >&2; return 1
        fi
        if [[ -f "${log_file}" ]] && grep -qE "${pattern}" "${log_file}" 2>/dev/null; then
            echo "${label} - gate reached"; return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for ${label}" >&2; return 1
}

declare -a BACKEND_PIDS=()
declare -a BACKEND_PORTS=()
declare -a BACKEND_LOGS=()

for i in $(seq 1 "${NUM_WORKERS}"); do
    if [[ "${SINGLE_GPU}" == "1" ]]; then
        gpu_id=0
    else
        gpu_id=$((i - 1))
    fi
    sys_var="VLLM${i}_SYSTEM_PORT";        sp="${!sys_var:-}";  system_port="${sp:-$((18079 + i * 2))}"
    kv_var="KV_EVENT_PORT_${i}";           kp="${!kv_var:-}";   kv_port="${kp:-$((20079 + i))}"
    name_var="VLLM${i}_SERVED_MODEL_NAME"; sn="${!name_var:-}"; served_name="${sn:-${MODEL}__internal}"
    log_file="${BACKEND_LOG_DIR}/worker_${i}.log"

    echo "=== Starting vLLM backend worker #${i} (GPU ${gpu_id}) ==="
    env "${COMMON_ENV[@]}" \
        "CUDA_VISIBLE_DEVICES=${gpu_id}" \
        "DYN_SYSTEM_PORT=${system_port}" \
        "DYN_VLLM_KV_EVENT_PORT=${kv_port}" \
        "${PYTHON_BIN}" -m dynamo.vllm \
            --model "${MODEL}" \
            --served-model-name "${served_name}" \
            --enable-prefix-caching \
            --enable-multimodal \
            --block-size "${BLOCK_SIZE}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            ${VLLM_EXTRA_ARGS} > "${log_file}" 2>&1 &
    PIDS+=($!)
    BACKEND_PIDS+=($!)
    BACKEND_PORTS+=("${system_port}")
    BACKEND_LOGS+=("${log_file}")

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

echo "=== Waiting for all ${NUM_WORKERS} workers to be ready ==="
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    wait_ready "http://127.0.0.1:${BACKEND_PORTS[$i]}/health" "vLLM backend #$((i + 1))" 900
done
echo "All backend workers are ready."

echo "=== Starting vLLM MM Router Worker ==="
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}" \
    "MM_ROUTER_REWRITE_IMAGE_URLS=${MM_ROUTER_REWRITE_IMAGE_URLS}" \
    'DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]' \
    "${PYTHON_BIN}" -m examples.backends.vllm.mm_router_worker \
        --model "${MODEL}" \
        --namespace "${NAMESPACE}" \
        --component "${MM_ROUTER_COMPONENT}" \
        --endpoint generate \
        --downstream-component "${BACKEND_COMPONENT}" \
        --downstream-endpoint generate \
        --block-size "${BLOCK_SIZE}" \
        ${MM_ROUTER_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${MM_ROUTER_SYSTEM_PORT}/health" "MM router" 300

echo
echo "=== Starting frontend ==="
env "${COMMON_ENV[@]}" \
    "PYTHONPATH=${DYNAMO_ROOT}/components/src" \
    "${PYTHON_BIN}" -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode round-robin \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend_models "http://127.0.0.1:${HTTP_PORT}/v1/models" 300

echo
echo "=== All services are ready ==="
echo "Frontend:  http://127.0.0.1:${HTTP_PORT}"
echo "MM Router: http://127.0.0.1:${MM_ROUTER_SYSTEM_PORT}/health"
for i in $(seq 1 "${NUM_WORKERS}"); do
    sys_var="VLLM${i}_SYSTEM_PORT"; sp="${!sys_var:-$((18079 + i * 2))}"
    echo "vLLM backend${i}: http://127.0.0.1:${sp}/health"
done
echo
echo "Press Ctrl+C to stop all services"

wait
