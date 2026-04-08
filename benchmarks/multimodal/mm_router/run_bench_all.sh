#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Master benchmark runner: runs all 3 scenarios sequentially.
# Each scenario restarts the server for each concurrency level.
#
# Scenarios:
#   frontend : NEW — Frontend vLLM processor + KvRouter + NIXL
#   mm       : OLD — Frontend -> MM Router Worker -> Backend
#   rr       : BASELINE — Frontend (round-robin) -> Backend
#
# Usage:
#   bash run_bench_all.sh                    # all scenarios, conc 1,4,8
#   CONC_LEVELS=1,4 bash run_bench_all.sh    # override concurrency
#   SCENARIOS=frontend bash run_bench_all.sh # single scenario
#
# Status files in LOG_DIR:
#   {name}.status   -> PENDING | RUNNING | FAILED | SKIPPED | DONE
#   bench.status    -> ALL_DONE when everything finishes
#   bench_master.log -> timestamped event log

set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${BENCH_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Scenario definitions: name:launch_script
ALL_SCENARIOS=(
    "frontend:launch_frontend_routing.sh"
    "mm:launch_mm_router_worker.sh"
    "rr:launch_rr_baseline.sh"
)

# Allow running a subset via SCENARIOS env var (comma-separated)
if [[ -n "${SCENARIOS:-}" ]]; then
    IFS=',' read -ra SELECTED <<< "${SCENARIOS}"
    FILTERED=()
    for entry in "${ALL_SCENARIOS[@]}"; do
        name="${entry%%:*}"
        for sel in "${SELECTED[@]}"; do
            if [[ "${name}" == "${sel}" ]]; then
                FILTERED+=("${entry}")
            fi
        done
    done
    ALL_SCENARIOS=("${FILTERED[@]}")
fi

BENCH_ENV=(
    "SINGLE_GPU=${SINGLE_GPU:-0}"
    "NUM_WORKERS=${NUM_WORKERS:-8}"
    "MODEL=${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}"
    "MAX_MODEL_LEN=${MAX_MODEL_LEN:-100426}"
    "CONC_LEVELS=${CONC_LEVELS:-1,4,8}"
    "PYTHONHASHSEED=0"
)

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/bench_master.log"
}

echo "ALL_PENDING" > "${LOG_DIR}/bench.status"
log "=== Benchmark master runner started ==="
log "Scenarios: ${ALL_SCENARIOS[*]}"
log "Env: ${BENCH_ENV[*]}"

for entry in "${ALL_SCENARIOS[@]}"; do
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
log "Results in: ${LOG_DIR}"
log ""
log "Result directories:"
log "  frontend: ls ${LOG_DIR}/frontend_*"
log "  mm:       ls ${LOG_DIR}/mm_*"
log "  rr:       ls ${LOG_DIR}/rr_*"
