#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark launch script: MM-aware KV routing via Rust frontend (no mm_router_worker,
# no --dyn-chat-processor vllm).
#
# Architecture:
#   Frontend (Rust preprocessor, --router-mode kv) --> N vLLM backends
#
# Key differences from launch_benchmark_no_mm_worker.sh:
#   - Workers run with --frontend-decoding: the Rust frontend downloads, decodes images,
#     and transfers pixel data to the worker via NIXL RDMA. multimodal_config is
#     auto-detected from the model's preprocessor_config.json (Qwen2VLImageProcessor family).
#   - Frontend does NOT use --dyn-chat-processor vllm: the Rust preprocessor handles
#     token expansion (~fast, dimension-based) and mm_hash (blake3) instead of the
#     Python vLLM processor (~16ms). mm_routing_info is built in Rust and passed
#     directly to the KV router.
#
# Usage:
#   ./launch_rust_frontend_mm.sh
#
#   # Override defaults:
#   NUM_WORKERS=1 MODEL=Qwen/Qwen2.5-VL-7B-Instruct ./launch_rust_frontend_mm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-/workspace}"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration — defaults match launch_benchmark_no_mm_worker.sh
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# Set SINGLE_GPU=1 to pin all workers to GPU 0 (single-GPU benchmarking).
# Default (0): each worker gets its own GPU (worker i -> GPU i-1).
SINGLE_GPU="${SINGLE_GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-100426}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

BACKEND_COMPONENT="${BACKEND_COMPONENT:-backend}"

VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

# Set deterministic hash seed for consistent KV block hashes
export PYTHONHASHSEED=0

echo "=== Rust Frontend MM: KV routing (${NUM_WORKERS} workers) ==="
echo "Architecture: Frontend(Rust+kv) --> ${NUM_WORKERS}x vLLM backend [--frontend-decoding, no mm_router_worker]"
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
    echo "  worker #${i}: system_port=${sp} kv_event_port=${kp} GPU=$((i-1))"
done
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
    local timeout_s="${3:-900}"
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
echo "  docker compose -f deploy/docker-compose.yml up -d"
echo

PYTHONPATH_VALUE="${DYNAMO_ROOT}/components/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

# ---------------------------------------------------------------------------
# vLLM backend workers (one per GPU)
# --frontend-decoding enables:
#   1. media_decoder in the MDC → Rust frontend downloads + decodes images
#   2. NIXL RDMA pixel transfer from frontend to worker
#   3. auto-detection of multimodal_config (patch_size/merge_size/image_token_id)
#      from preprocessor_config.json at MDC build time
# ---------------------------------------------------------------------------
for i in $(seq 1 "${NUM_WORKERS}"); do
    if [[ "${SINGLE_GPU}" == "1" ]]; then
        gpu_id=0
    else
        gpu_id=$((i - 1))
    fi
    sys_var="VLLM${i}_SYSTEM_PORT"; sp="${!sys_var:-}"; system_port="${sp:-$((18079 + i * 2))}"
    kv_var="KV_EVENT_PORT_${i}";   kp="${!kv_var:-}"; kv_port="${kp:-$((20079 + i))}"

    echo "=== Starting vLLM backend worker #${i} (GPU ${gpu_id}, port ${system_port}) ==="
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    env "${COMMON_ENV[@]}" \
        "DYN_SYSTEM_PORT=${system_port}" \
        "DYN_VLLM_KV_EVENT_PORT=${kv_port}" \
        "${PYTHON_BIN}" -m dynamo.vllm \
            --model "${MODEL}" \
            --enable-prefix-caching \
            --enable-multimodal \
            --frontend-decoding \
            --block-size "${BLOCK_SIZE}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            ${VLLM_EXTRA_ARGS} &
    PIDS+=($!)

    wait_ready "http://127.0.0.1:${system_port}/health" "vLLM backend #${i}" 900
    echo
done

# ---------------------------------------------------------------------------
# Frontend: Rust preprocessor + KV router
#
# No --dyn-chat-processor vllm: the Rust preprocessor handles everything.
# multimodal_config is carried in the MDC (auto-detected above) and used by
# the Rust preprocessor to:
#   - expand image placeholder tokens from dimensions (fast, ~sub-ms)
#   - compute mm_hash from decoded pixel bytes (blake3)
#   - build block_mm_infos and pass mm_routing_info to the KV router
# ---------------------------------------------------------------------------
echo "=== Starting frontend (Rust preprocessor, --router-mode kv) ==="
env "${COMMON_ENV[@]}" \
    "PYTHONPATH=${PYTHONPATH_VALUE}" \
    "${PYTHON_BIN}" -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode kv \
        --kv-cache-block-size "${BLOCK_SIZE}" \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend_models "http://127.0.0.1:${HTTP_PORT}/v1/models" 300

echo
echo "=== All services are ready ==="
echo "Frontend:  http://127.0.0.1:${HTTP_PORT}"
for i in $(seq 1 "${NUM_WORKERS}"); do
    sys_var="VLLM${i}_SYSTEM_PORT"; sp="${!sys_var:-$((18079 + i * 2))}"
    echo "vLLM backend #${i}: http://127.0.0.1:${sp}/health"
done
echo
echo "To verify multimodal_config was auto-detected, check worker logs for:"
echo "  Auto-detected QwenVL multimodal config"
echo
echo "Press Ctrl+C to stop all services"

wait
