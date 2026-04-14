#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a single benchmark scenario: start server, sweep conc=N, restart, sweep conc=M, ...
# Server is restarted between each concurrency level to avoid cache pollution.
#
# Usage:
#   bash run_scenario.sh <launch_script> <router_name> [conc_levels]
#
# Example:
#   bash run_scenario.sh launch_frontend_routing.sh frontend 1,4,8

set -euo pipefail

LAUNCH_SCRIPT="${1:?Usage: run_scenario.sh <launch_script> <router_name> [conc_levels]}"
ROUTER="${2:?Usage: run_scenario.sh <launch_script> <router_name> [conc_levels]}"
CONC_ARG="${3:-${CONC_LEVELS:-1,4,8}}"
IFS=',' read -ra CONC_LEVELS <<< "${CONC_ARG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUM_WORKERS="${NUM_WORKERS:-8}"
export MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE:-500}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-100426}"
export SINGLE_GPU="${SINGLE_GPU:-0}"
export PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
export NUM_FRONTENDS="${NUM_FRONTENDS:-1}"
export PYTHONHASHSEED=0

# frontend_pool scenario: enable preprocess worker pool
if [[ "${ROUTER}" == "frontend_pool" ]]; then
    export PREPROCESS_WORKERS="${PREPROCESS_WORKERS_COUNT:-4}"
fi

# Build AIPERF_URL from NUM_FRONTENDS
HTTP_PORT="${HTTP_PORT:-8000}"
AIPERF_URLS=""
for f in $(seq 1 "${NUM_FRONTENDS}"); do
    FE_PORT=$((HTTP_PORT + f - 1))
    AIPERF_URLS="${AIPERF_URLS} http://127.0.0.1:${FE_PORT}"
done
export AIPERF_URL="${AIPERF_URLS# }"  # trim leading space

DATASET_DIR="${SCRIPT_DIR}/datasets"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
HTTP_PORT="${HTTP_PORT:-8000}"

echo "============================================================"
echo "  Scenario    : ${ROUTER}"
echo "  Script      : ${LAUNCH_SCRIPT}"
echo "  Model       : ${MODEL}"
echo "  Workers     : ${NUM_WORKERS}"
echo "  Single GPU  : ${SINGLE_GPU}"
echo "  Conc levels : ${CONC_LEVELS[*]}"
echo "============================================================"

start_server() {
    echo "[$(date '+%H:%M:%S')] Starting server: ${LAUNCH_SCRIPT} ..."
    bash "${LAUNCH_SCRIPT}" &
    SERVER_PID=$!
    echo "[$(date '+%H:%M:%S')] Server PID: ${SERVER_PID}"

    # Wait for /v1/models to respond (HTTP server up)
    local timeout_s=900
    local deadline=$((SECONDS + timeout_s))
    echo "[$(date '+%H:%M:%S')] Waiting for frontend at http://127.0.0.1:${HTTP_PORT}/v1/models ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" >/dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] Frontend HTTP is up."
            break
        fi
        sleep 2
    done
    if (( SECONDS >= deadline )); then
        echo "[$(date '+%H:%M:%S')] ERROR: Server did not become ready within ${timeout_s}s" >&2
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi

    # Wait for processor to be fully ready (can serve a real request)
    echo "[$(date '+%H:%M:%S')] Waiting for processor to initialize (sending test request)..."
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
            -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            echo "[$(date '+%H:%M:%S')] Server is fully ready."
            return 0
        fi
        sleep 2
    done
    echo "[$(date '+%H:%M:%S')] WARNING: Processor may not be fully ready (timed out on test request)" >&2
}

stop_server() {
    echo "[$(date '+%H:%M:%S')] Stopping server (PID ${SERVER_PID}) ..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Waiting 15s for ports to free ..."
    sleep 15
    echo "[$(date '+%H:%M:%S')] Server stopped."
}

run_sweep() {
    local conc="$1"
    echo "[$(date '+%H:%M:%S')] Running sweep conc=${conc} router=${ROUTER} ..."
    bash run_sweep.sh \
        --model "${MODEL}" \
        --router "${ROUTER}" \
        --dataset-dir "${DATASET_DIR}" \
        --log-dir "${LOG_DIR}" \
        --conc "${conc}"
    echo "[$(date '+%H:%M:%S')] Sweep conc=${conc} done."
}

mkdir -p "${LOG_DIR}"

for conc in "${CONC_LEVELS[@]}"; do
    # Scale image cache to cover pool_50 (conc*60 unique images) plus buffer.
    computed_cache=$(( conc * 70 ))
    if (( computed_cache < 500 )); then computed_cache=500; fi
    export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE_OVERRIDE:-${computed_cache}}"
    echo "[$(date '+%H:%M:%S')] conc=${conc} -> DYN_MM_IMAGE_CACHE_SIZE=${DYN_MM_IMAGE_CACHE_SIZE}"
    start_server
    run_sweep "${conc}"
    stop_server
done

echo "============================================================"
echo "  Scenario ${ROUTER} COMPLETE"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"
