#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Master benchmark runner: runs all 4 scenarios sequentially.
# Writes status files so a monitor agent can track progress and trigger retries.
#
# Status files in LOG_DIR:
#   {name}.status   -> PENDING | RUNNING | FAILED | SKIPPED | DONE
#   bench.status    -> ALL_DONE when everything finishes
#   {name}.retry    -> written by monitor agent to trigger a retry
#   bench_master.log -> timestamped event log

set -uo pipefail

BENCH_DIR=/workspace/benchmarks/multimodal/mm_router
LOG_DIR="${BENCH_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCENARIOS=(
    "rr:launch_workers_rr_baseline.sh"
    "mm:launch_workers.sh"
    "vllm-processor:launch_vllm_processor.sh"
    "rust:launch_rust_frontend_mm.sh"
)

# All values can be overridden by setting env vars before running this script.
BENCH_ENV=(
    "SINGLE_GPU=${SINGLE_GPU:-0}"
    "NUM_WORKERS=${NUM_WORKERS:-8}"
    "MODEL=${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}"
    "MAX_MODEL_LEN=${MAX_MODEL_LEN:-100426}"
    "DYN_MM_IMAGE_CACHE_SIZE=${DYN_MM_IMAGE_CACHE_SIZE:-500}"
    "DYN_MM_MEDIA_CACHE_SIZE=${DYN_MM_MEDIA_CACHE_SIZE:-500}"
    "PYTHONHASHSEED=0"
)

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/bench_master.log"
}

echo "ALL_PENDING" > "${LOG_DIR}/bench.status"
log "=== Benchmark master runner started ==="
log "Scenarios: ${SCENARIOS[*]}"

for entry in "${SCENARIOS[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"

    echo "PENDING" > "${LOG_DIR}/${name}.status"
    retry=0
    max_retries=2

    while [[ $retry -le $max_retries ]]; do
        log "--- Scenario: ${name} | attempt $((retry + 1)) ---"
        echo "RUNNING" > "${LOG_DIR}/${name}.status"
        rm -f "${LOG_DIR}/${name}.retry"

        cd "${BENCH_DIR}"
        env "${BENCH_ENV[@]}" bash run_scenario.sh "${script}" "${name}" \
            2>&1 | tee "${LOG_DIR}/run_${name}.log"
        rc=${PIPESTATUS[0]}

        if [[ $rc -eq 0 ]]; then
            echo "DONE" > "${LOG_DIR}/${name}.status"
            log "Scenario ${name} DONE."
            break
        fi

        # Scenario failed
        echo "FAILED" > "${LOG_DIR}/${name}.status"
        log "Scenario ${name} FAILED (exit ${rc})."

        if [[ $retry -ge $max_retries ]]; then
            log "Max retries reached for ${name}, skipping."
            echo "SKIPPED" > "${LOG_DIR}/${name}.status"
            break
        fi

        log "Waiting up to 10 minutes for retry signal (${LOG_DIR}/${name}.retry) ..."
        waited=0
        got_retry=0
        while [[ $waited -lt 600 ]]; do
            if [[ -f "${LOG_DIR}/${name}.retry" ]]; then
                log "Retry signal received for ${name}."
                got_retry=1
                ((retry++))
                break
            fi
            sleep 15
            ((waited += 15))
        done

        if [[ $got_retry -eq 0 ]]; then
            log "No retry signal within timeout for ${name}, skipping."
            echo "SKIPPED" > "${LOG_DIR}/${name}.status"
            break
        fi
    done
done

echo "ALL_DONE" > "${LOG_DIR}/bench.status"
log "=== All benchmark scenarios complete ==="
