#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a single benchmark scenario: start server, sweep conc=1, restart server, sweep conc=4, kill.
#
# Usage:
#   bash run_scenario.sh <launch_script> <router_name>
#
# Example:
#   bash run_scenario.sh launch_workers_rr_baseline.sh rr

set -euo pipefail

LAUNCH_SCRIPT="${1:?Usage: run_scenario.sh <launch_script> <router_name> [conc_levels]}"
ROUTER="${2:?Usage: run_scenario.sh <launch_script> <router_name> [conc_levels]}"
# Optional 3rd arg: comma-separated concurrency levels, e.g. "1,4,8,16,32,64"
# Falls back to CONC_LEVELS env var, then default "1,4".
CONC_ARG="${3:-${CONC_LEVELS:-1,4}}"
IFS=',' read -ra CONC_LEVELS <<< "${CONC_ARG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export NUM_WORKERS="${NUM_WORKERS:-2}"
export MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE:-500}"
export DYN_MM_MEDIA_CACHE_SIZE="${DYN_MM_MEDIA_CACHE_SIZE:-500}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export PYTHONHASHSEED=0

DATASET_DIR="${SCRIPT_DIR}/datasets"
LOG_DIR="${SCRIPT_DIR}/logs"
HTTP_PORT="${HTTP_PORT:-8000}"

echo "============================================================"
echo "  Scenario    : ${ROUTER}"
echo "  Script      : ${LAUNCH_SCRIPT}"
echo "  Model       : ${MODEL}"
echo "  Workers     : ${NUM_WORKERS}"
echo "  Conc levels : ${CONC_LEVELS[*]}"
echo "============================================================"

start_server() {
    echo "[$(date '+%H:%M:%S')] Starting server: ${LAUNCH_SCRIPT} ..."
    bash "${LAUNCH_SCRIPT}" &
    SERVER_PID=$!
    echo "[$(date '+%H:%M:%S')] Server PID: ${SERVER_PID}"

    # Wait until frontend models endpoint responds
    local timeout_s=900
    local deadline=$((SECONDS + timeout_s))
    echo "[$(date '+%H:%M:%S')] Waiting for frontend at http://127.0.0.1:${HTTP_PORT}/v1/models ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" >/dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] Server is ready."
            return 0
        fi
        sleep 2
    done
    echo "[$(date '+%H:%M:%S')] ERROR: Server did not become ready within ${timeout_s}s" >&2
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
}

stop_server() {
    echo "[$(date '+%H:%M:%S')] Stopping server (PID ${SERVER_PID}) ..."
    kill "${SERVER_PID}" 2>/dev/null || true
    # Wait for all child processes to exit
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
    # Scale image cache to cover pool_50 (conc*60 unique images) plus ~17% buffer.
    # Formula: max(500, conc * 70). Matches empirical values: conc=4→500, conc=32→2240, conc=64→4480.
    computed_cache=$(( conc * 70 ))
    if (( computed_cache < 500 )); then computed_cache=500; fi
    export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE_OVERRIDE:-${computed_cache}}"
    export DYN_MM_MEDIA_CACHE_SIZE="${DYN_MM_MEDIA_CACHE_SIZE_OVERRIDE:-${computed_cache}}"
    echo "[$(date '+%H:%M:%S')] conc=${conc} -> DYN_MM_IMAGE_CACHE_SIZE=${DYN_MM_IMAGE_CACHE_SIZE} DYN_MM_MEDIA_CACHE_SIZE=${DYN_MM_MEDIA_CACHE_SIZE}"
    start_server
    run_sweep "${conc}"
    stop_server
done

echo "============================================================"
echo "  Scenario ${ROUTER} COMPLETE"
echo "  Logs: ${LOG_DIR}"
echo "============================================================"
