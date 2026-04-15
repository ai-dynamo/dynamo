#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Automated nsys profiling comparison across all routing architectures.
#
# Scenarios:
#   frontend_shm : Frontend (vLLM processor + KvRouter + SHM transfer)  — upstream/dynamo
#   frontend_nixl: Frontend (vLLM processor + KvRouter + NIXL transfer) — upstream/dynamo
#   rr           : Round-robin baseline (no MM routing)                 — upstream/dynamo
#   mm           : MM Router Worker (round-robin + KvRouter)            — dynamo/
#
# Each case: launch server under nsys, warmup, send measured requests, stop.
#
# Usage:
#   bash run_nsys_compare.sh
#   CONC_LEVELS=1 bash run_nsys_compare.sh
#   SCENARIOS=frontend_shm,mm bash run_nsys_compare.sh
#   SCENARIOS=rr CONC_LEVELS=1 bash run_nsys_compare.sh
#
# Output: ./nsys_compare/
#   frontend_shm_conc1.nsys-rep, frontend_nixl_conc1.nsys-rep
#   mm_conc1.nsys-rep, rr_conc1.nsys-rep

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# The investigation repo root (contains dynamo/ and upstream/dynamo/).
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
# upstream/dynamo is used for RR + frontend routing (NIXL/SHM).
UPSTREAM_DYNAMO_ROOT="${UPSTREAM_DYNAMO_ROOT:-${REPO_ROOT}/upstream/dynamo}"
# dynamo/ is used for MM router worker.
DYNAMO_ROOT_MM="${DYNAMO_ROOT_MM:-${REPO_ROOT}/dynamo}"

export NUM_WORKERS="${NUM_WORKERS:-2}"
export MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export DYN_MM_IMAGE_CACHE_SIZE="${DYN_MM_IMAGE_CACHE_SIZE:-500}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export SINGLE_GPU="${SINGLE_GPU:-1}"
export DYN_NVTX=1
export PYTHONHASHSEED=0

HTTP_PORT="${HTTP_PORT:-8000}"
NSYS_DIR="${NSYS_DIR:-${SCRIPT_DIR}/nsys_compare}"
DATASET_DIR="${SCRIPT_DIR}/datasets"

WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
MEASURED_REQUESTS="${MEASURED_REQUESTS:-25}"
NSYS_DELAY="${NSYS_DELAY:-60}"

# Parse scenarios and conc levels
IFS=',' read -ra CONC_LEVELS <<< "${CONC_LEVELS:-1,8}"
IFS=',' read -ra SCENARIOS <<< "${SCENARIOS:-frontend_shm,mm}"

mkdir -p "${NSYS_DIR}"

SERVER_PID=""

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        log "Stopping server (PID ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
        log "Waiting 15s for ports to free..."
        sleep 15
        log "Server stopped."
    fi
}

trap stop_server EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + 900))
    log "Waiting for frontend at http://127.0.0.1:${HTTP_PORT}/v1/models ..."
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" >/dev/null 2>&1; then
            log "Frontend HTTP is up."
            break
        fi
        sleep 2
    done

    # Wait for processor to be ready
    log "Waiting for processor warmup (test request)..."
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
            -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            log "Server fully ready."
            return 0
        fi
        sleep 2
    done
    log "WARNING: Server may not be fully ready."
}

run_warmup() {
    local conc="$1"
    log "Sending ${WARMUP_REQUESTS} warmup requests (conc=${conc})..."

    # Pick dataset based on conc
    local warmup_file
    if [[ "${conc}" -le 1 ]]; then
        warmup_file="${DATASET_DIR}/100req_1img_10pool_datauri.jsonl"
    else
        warmup_file="${DATASET_DIR}/800req_1img_80pool_datauri.jsonl"
    fi

    if [[ ! -f "${warmup_file}" ]]; then
        warmup_file="${DATASET_DIR}/warmup_50req_1img_5pool_http.jsonl"
    fi

    if [[ -f "${warmup_file}" ]]; then
        aiperf profile \
            --model "${MODEL}" \
            --input-file "${warmup_file}" \
            --custom-dataset-type single_turn \
            --osl 1 \
            --request-count "${WARMUP_REQUESTS}" \
            --concurrency "${conc}" \
            --artifact-dir "${NSYS_DIR}/_warmup_tmp" 2>&1 | tail -5
        rm -rf "${NSYS_DIR}/_warmup_tmp"
    fi
    log "Warmup complete."
}

run_measured() {
    local conc="$1"
    local label="$2"

    # Pick dataset
    local req_file
    if [[ "${conc}" -le 1 ]]; then
        req_file="${DATASET_DIR}/100req_1img_10pool_datauri.jsonl"
    else
        req_file="${DATASET_DIR}/800req_1img_80pool_datauri.jsonl"
    fi

    if [[ ! -f "${req_file}" ]]; then
        log "ERROR: Dataset not found: ${req_file}"
        log "Run: bash generate_datasets.sh ./datasets 1,8"
        return 1
    fi

    local out_dir="${NSYS_DIR}/${label}_conc${conc}_results"
    rm -rf "${out_dir}"

    log "Sending ${MEASURED_REQUESTS} measured requests (conc=${conc})..."
    aiperf profile \
        --model "${MODEL}" \
        --input-file "${req_file}" \
        --custom-dataset-type single_turn \
        --osl 1 \
        --request-count "${MEASURED_REQUESTS}" \
        --concurrency "${conc}" \
        --artifact-dir "${out_dir}" 2>&1 | tail -10
    log "Measured requests complete: ${out_dir}"
}

# Map scenario -> (launch_script, DYNAMO_ROOT, extra env vars).
# Writes to global vars: _LAUNCH_SCRIPT, _LAUNCH_DYNAMO_ROOT, _LAUNCH_ENV
get_launch_config() {
    _LAUNCH_ENV=()
    case "$1" in
        frontend_shm)
            _LAUNCH_SCRIPT="launch_frontend_routing.sh"
            _LAUNCH_DYNAMO_ROOT="${UPSTREAM_DYNAMO_ROOT}"
            _LAUNCH_ENV=("DYNAMO_MM_TRANSFER=shm")
            ;;
        frontend_nixl)
            _LAUNCH_SCRIPT="launch_frontend_routing.sh"
            _LAUNCH_DYNAMO_ROOT="${UPSTREAM_DYNAMO_ROOT}"
            _LAUNCH_ENV=()  # NIXL is default
            ;;
        frontend)
            # Alias: defaults to SHM
            _LAUNCH_SCRIPT="launch_frontend_routing.sh"
            _LAUNCH_DYNAMO_ROOT="${UPSTREAM_DYNAMO_ROOT}"
            _LAUNCH_ENV=("DYNAMO_MM_TRANSFER=shm")
            ;;
        rr)
            _LAUNCH_SCRIPT="launch_rr_baseline.sh"
            _LAUNCH_DYNAMO_ROOT="${UPSTREAM_DYNAMO_ROOT}"
            ;;
        mm)
            _LAUNCH_SCRIPT="launch_mm_router_worker.sh"
            _LAUNCH_DYNAMO_ROOT="${DYNAMO_ROOT_MM}"
            ;;
        *)
            _LAUNCH_SCRIPT=""
            return 1
            ;;
    esac
}

echo "============================================================"
echo "  NVTX Profiling Comparison"
echo "  Model            : ${MODEL}"
echo "  Workers          : ${NUM_WORKERS}"
echo "  Scenarios        : ${SCENARIOS[*]}"
echo "  Conc levels      : ${CONC_LEVELS[*]}"
echo "  Warmup           : ${WARMUP_REQUESTS} requests"
echo "  Measured         : ${MEASURED_REQUESTS} requests"
echo "  nsys delay       : ${NSYS_DELAY}s"
echo "  Upstream dynamo  : ${UPSTREAM_DYNAMO_ROOT}"
echo "  MM dynamo        : ${DYNAMO_ROOT_MM}"
echo "  Output           : ${NSYS_DIR}/"
echo "============================================================"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    get_launch_config "${scenario}" || {
        log "Unknown scenario: ${scenario}, skipping."
        continue
    }

    for conc in "${CONC_LEVELS[@]}"; do
        trace_name="${scenario}_conc${conc}"
        trace_file="${NSYS_DIR}/${trace_name}"

        if [[ -f "${trace_file}.nsys-rep" ]]; then
            log "SKIP: ${trace_file}.nsys-rep already exists. Delete to re-run."
            continue
        fi

        log "========== ${scenario} conc=${conc} =========="
        log "  Launch: ${_LAUNCH_SCRIPT}"
        log "  DYNAMO_ROOT: ${_LAUNCH_DYNAMO_ROOT}"
        [[ ${#_LAUNCH_ENV[@]} -gt 0 ]] && log "  Extra env: ${_LAUNCH_ENV[*]}"

        # Launch server under nsys with the correct DYNAMO_ROOT
        log "Starting server under nsys (delay=${NSYS_DELAY}s)..."
        env \
            "DYNAMO_ROOT=${_LAUNCH_DYNAMO_ROOT}" \
            "${_LAUNCH_ENV[@]+"${_LAUNCH_ENV[@]}"}" \
        nsys profile \
            -t cuda,nvtx,osrt \
            --cuda-memory-usage=true \
            --python-backtrace=cuda \
            --python-sampling=true \
            --trace-fork-before-exec=true \
            --delay="${NSYS_DELAY}" \
            --wait=all \
            -o "${trace_file}" \
            bash "${_LAUNCH_SCRIPT}" &
        SERVER_PID=$!
        log "nsys+server PID: ${SERVER_PID}"

        # Wait for server to be ready
        wait_for_server

        # Warmup (before nsys delay window to avoid capturing warmup)
        run_warmup "${conc}"
        sleep 3

        # Measured requests (nsys should be collecting by now)
        run_measured "${conc}" "${scenario}"

        # Stop server (nsys will finalize the trace)
        stop_server

        log "Trace saved: ${trace_file}.nsys-rep"
        echo ""
    done
done

log "============================================================"
log "  All traces complete!"
log "  Output: ${NSYS_DIR}/"
log ""
log "  Analyze with:"
for scenario in "${SCENARIOS[@]}"; do
    for conc in "${CONC_LEVELS[@]}"; do
        log "    nsys stats --report nvtx_pushpop_trace ${NSYS_DIR}/${scenario}_conc${conc}.nsys-rep --force-export=true 2>&1 | grep 'dynamo:mm_'"
    done
done
log "============================================================"
