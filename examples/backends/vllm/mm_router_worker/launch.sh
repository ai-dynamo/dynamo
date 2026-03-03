#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for vLLM MM Router Worker demo:
#   Frontend (round-robin) -> MM Router Worker -> vLLM backend
#
# This script is intended as a step-by-step runnable demo on a single machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
NAMESPACE="${NAMESPACE:-dynamo}"
HTTP_PORT="${HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"            # Must match vLLM backend KV block size
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"

VLLM_SYSTEM_PORT="${VLLM_SYSTEM_PORT:-18081}"
MM_ROUTER_SYSTEM_PORT="${MM_ROUTER_SYSTEM_PORT:-18082}"

MM_ROUTER_COMPONENT="${MM_ROUTER_COMPONENT:-mm_router}"
BACKEND_COMPONENT="${BACKEND_COMPONENT:-backend}"  # dynamo.vllm default

# Extra args (word-splitting is intentional for shell-style overrides)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"
MM_ROUTER_EXTRA_ARGS="${MM_ROUTER_EXTRA_ARGS:-}"

echo "=== vLLM MM Router Worker Launch Script ==="
echo "Working directory: ${DYNAMO_ROOT}"
echo "MODEL=${MODEL}"
echo "NAMESPACE=${NAMESPACE}"
echo "HTTP_PORT=${HTTP_PORT}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "VLLM_SYSTEM_PORT=${VLLM_SYSTEM_PORT}"
echo "MM_ROUTER_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}"
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
    "DYN_REQUEST_PLANE=nats"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
)

echo
echo "=== Starting vLLM backend worker ==="
# Use an internal served-model-name so frontend traffic goes to the MM router
# (which registers the public model name) instead of directly to the backend.
env "${COMMON_ENV[@]}" \
    "DYN_SYSTEM_PORT=${VLLM_SYSTEM_PORT}" \
    python -m dynamo.vllm \
        --model "${MODEL}" \
        --enable-multimodal \
        --block-size "${BLOCK_SIZE}" \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --served-model-name "${MODEL}__internal" \
        ${VLLM_EXTRA_ARGS} &
PIDS+=($!)

wait_ready "http://127.0.0.1:${VLLM_SYSTEM_PORT}/health" "vLLM backend" 900

echo
echo "=== Starting vLLM MM Router Worker ==="
env "${COMMON_ENV[@]}" \
    "DYN_LOG=debug" \
    "DYN_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT}" \
    'DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]' \
    python -m examples.backends.vllm.mm_router_worker \
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
    "DYN_LOG=info" \
    python -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode round-robin \
        ${FRONTEND_EXTRA_ARGS} &
PIDS+=($!)

wait_frontend_models "http://127.0.0.1:${HTTP_PORT}/v1/models" 300

echo
echo "=== All services are ready ==="
echo "Frontend:    http://127.0.0.1:${HTTP_PORT}"
echo "MM Router:   http://127.0.0.1:${MM_ROUTER_SYSTEM_PORT}/health"
echo "vLLM backend:http://127.0.0.1:${VLLM_SYSTEM_PORT}/health"
echo
echo "Try the same multimodal request twice and compare MM router logs for:"
echo '  [ROUTING] Best: worker_... with X/Y blocks overlap'
echo
echo "Example:"
echo "  curl http://127.0.0.1:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"http://images.cocodataset.org/test2017/000000000001.jpg\"}}]}],\"max_tokens\":32}'"
echo
echo "Press Ctrl+C to stop all services"

wait
