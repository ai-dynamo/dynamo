#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for a local BASELINE (no MM router) with TWO vLLM backend workers:
#   Frontend (round-robin) -> vLLM backend #1/#2
#
# This script assumes etcd + NATS are already running (see deploy/docker-compose.yml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

# Defaults tuned for single ~48 GB GPU running two Qwen3-VL-2B workers.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

VLLM1_SYSTEM_PORT="${VLLM1_SYSTEM_PORT:-18081}"
VLLM2_SYSTEM_PORT="${VLLM2_SYSTEM_PORT:-18083}"

KV_EVENT_PORT_1="${KV_EVENT_PORT_1:-20080}"
KV_EVENT_PORT_2="${KV_EVENT_PORT_2:-20081}"

# Keep the same variable shape as launch_two_workers.sh for easier A/B compare.
# Baseline defaults both to MODEL so frontend can request one model name directly.
VLLM1_SERVED_MODEL_NAME="${VLLM1_SERVED_MODEL_NAME:-${MODEL}}"
VLLM2_SERVED_MODEL_NAME="${VLLM2_SERVED_MODEL_NAME:-${MODEL}}"

# Extra args (word-splitting intentional for shell-style overrides)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

echo "=== vLLM Round-Robin Baseline (Two Workers) ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL=${MODEL}"
echo "VLLM1_SERVED_MODEL_NAME=${VLLM1_SERVED_MODEL_NAME}"
echo "VLLM2_SERVED_MODEL_NAME=${VLLM2_SERVED_MODEL_NAME}"
echo "NAMESPACE=${NAMESPACE}"
echo "HTTP_PORT=${HTTP_PORT}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "VLLM1_SYSTEM_PORT=${VLLM1_SYSTEM_PORT} (KV events ${KV_EVENT_PORT_1})"
echo "VLLM2_SYSTEM_PORT=${VLLM2_SYSTEM_PORT} (KV events ${KV_EVENT_PORT_2})"
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
    "DYN_LOG=debug"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

echo "=== Starting vLLM backend worker #1 ==="
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${VLLM1_SYSTEM_PORT}" \
    "DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_1}" \
    "${PYTHON_BIN}" -m dynamo.vllm \
        --model "${MODEL}" \
        --served-model-name "${VLLM1_SERVED_MODEL_NAME}" \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        ${VLLM_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${VLLM1_SYSTEM_PORT}/health" "vLLM backend #1" 900

echo
echo "=== Starting vLLM backend worker #2 ==="
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${VLLM2_SYSTEM_PORT}" \
    "DYN_VLLM_KV_EVENT_PORT=${KV_EVENT_PORT_2}" \
    "${PYTHON_BIN}" -m dynamo.vllm \
        --model "${MODEL}" \
        --served-model-name "${VLLM2_SERVED_MODEL_NAME}" \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        ${VLLM_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${VLLM2_SYSTEM_PORT}/health" "vLLM backend #2" 900

echo
echo "=== Starting frontend (round-robin baseline) ==="
env "${COMMON_ENV[@]}" \
    "${PYTHON_BIN}" -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode round-robin \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend_models "http://127.0.0.1:${HTTP_PORT}/v1/models" 300

echo
echo "=== All services are ready ==="
echo "Frontend:      http://127.0.0.1:${HTTP_PORT}"
echo "vLLM backend1: http://127.0.0.1:${VLLM1_SYSTEM_PORT}/health"
echo "vLLM backend2: http://127.0.0.1:${VLLM2_SYSTEM_PORT}/health"
echo
echo "Test request:"
echo "  ./examples/backends/vllm/mm_router_worker/test_mm_request.sh"
echo
echo "Press Ctrl+C to stop all services"

wait
