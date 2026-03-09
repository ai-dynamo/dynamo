#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for a local MM-router demo with TWO vLLM backend workers:
#   Frontend (round-robin) -> MM Router Worker -> vLLM backend #1/#2
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
BLOCK_SIZE="${BLOCK_SIZE:-16}" # Must match vLLM backend KV block size

# Defaults tuned for single ~48 GB GPU running two Qwen3-VL-2B workers.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

VLLM1_SYSTEM_PORT="${VLLM1_SYSTEM_PORT:-18081}"
VLLM2_SYSTEM_PORT="${VLLM2_SYSTEM_PORT:-18083}"
MM_ROUTER_SYSTEM_PORT="${MM_ROUTER_SYSTEM_PORT:-18082}"

KV_EVENT_PORT_1="${KV_EVENT_PORT_1:-20080}"
KV_EVENT_PORT_2="${KV_EVENT_PORT_2:-20081}"

MM_ROUTER_COMPONENT="${MM_ROUTER_COMPONENT:-mm_router}"
BACKEND_COMPONENT="${BACKEND_COMPONENT:-backend}" # dynamo.vllm default
VLLM1_SERVED_MODEL_NAME="${VLLM1_SERVED_MODEL_NAME:-${MODEL}__internal_1}"
VLLM2_SERVED_MODEL_NAME="${VLLM2_SERVED_MODEL_NAME:-${MODEL}__internal_2}"

# Extra args (word-splitting intentional for shell-style overrides)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"
MM_ROUTER_EXTRA_ARGS="${MM_ROUTER_EXTRA_ARGS:-}"

echo "=== vLLM MM Router (Two Workers) Launch Script ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL=${MODEL}"
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
echo "MM_ROUTER_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}"
echo "MM_ROUTER_COMPONENT=${MM_ROUTER_COMPONENT}"
echo "BACKEND_COMPONENT=${BACKEND_COMPONENT}"
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
echo "=== Starting vLLM MM Router Worker ==="
env "${COMMON_ENV[@]}" \
    "PYTHONPATH=${PYTHONPATH_VALUE}" \
    "DYN_LOG=debug" \
    "DYN_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}" \
    "MM_ROUTER_IMAGE_TRANSPORT_MODE=${MM_ROUTER_IMAGE_TRANSPORT_MODE:-data_uri}" \
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
    "${PYTHON_BIN}" -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode round-robin \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend_models "http://127.0.0.1:${HTTP_PORT}/v1/models" 300

echo
echo "=== All services are ready ==="
echo "Frontend:      http://127.0.0.1:${HTTP_PORT}"
echo "MM Router:     http://127.0.0.1:${MM_ROUTER_SYSTEM_PORT}/health"
echo "vLLM backend1: http://127.0.0.1:${VLLM1_SYSTEM_PORT}/health"
echo "vLLM backend2: http://127.0.0.1:${VLLM2_SYSTEM_PORT}/health"
echo
echo "Test request (run twice):"
echo "  ./examples/backends/vllm/mm_router_worker/test_mm_request.sh"
echo
echo "Press Ctrl+C to stop all services"

wait
