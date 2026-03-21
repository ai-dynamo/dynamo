#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Concurrency sweep for MM Router vs RR Baseline vs No-MM-Worker vs Rust Frontend benchmarks.
#
# Runs aiperf for all combinations of:
#   - router: mm (MM Router), rr (RR Baseline), no-mm (no mm_router_worker, frontend KV routing),
#             rust (Rust frontend mm routing, no mm_router_worker, no --dyn-chat-processor vllm)
#   - transport: http, datauri
#   - concurrency: 1, 4, 8, 16, 32, 64
#   - pool: 10pool (~90% reuse), 60pool (~50% reuse), Npool (0% reuse)
#
# Request count = concurrency * 100.
#
# Output naming:
#   mm    : {N}w_{n}req_1img_{pool}pool_{mode}_conc{c}
#   rr    : rr_{N}w_{n}req_...
#   no-mm : no_mm_{N}w_{n}req_...
#   rust  : rust_{N}w_{n}req_...
#
# Usage:
#   bash run_sweep.sh --router mm     --dataset-dir ./datasets --log-dir ./logs
#   bash run_sweep.sh --router rr     --dataset-dir ./datasets --log-dir ./logs
#   bash run_sweep.sh --router no-mm  --dataset-dir ./datasets --log-dir ./logs
#   bash run_sweep.sh --router rust   --dataset-dir ./datasets --log-dir ./logs
#   bash run_sweep.sh --router all    --dataset-dir ./datasets --log-dir ./logs
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"
URL="${AIPERF_URL:-http://127.0.0.1:8000}"
ROUTER="${ROUTER:-all}"         # mm | rr | no-mm | rust | all
DATASET_DIR="${DATASET_DIR:-$(dirname "$0")/datasets}"
LOG_DIR="${LOG_DIR:-$(dirname "$0")/logs}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP_CONC="${WARMUP_CONC:-1}"

CONC_LEVELS=(1 4 8 16 32 64)   # overridden by --conc

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --router)       ROUTER="$2";      shift 2 ;;
        --dataset-dir)  DATASET_DIR="$2"; shift 2 ;;
        --log-dir)      LOG_DIR="$2";     shift 2 ;;
        --model)        MODEL="$2";       shift 2 ;;
        --url)          URL="$2";         shift 2 ;;
        --workers)      NUM_WORKERS="$2"; shift 2 ;;
        --conc)         IFS=',' read -ra CONC_LEVELS <<< "$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

echo "=== MM Concurrency Sweep ==="
echo "Model      : ${MODEL}"
echo "Router     : ${ROUTER}  (mm | rr | no-mm | rust | all)"
echo "Dataset dir: ${DATASET_DIR}"
echo "Log dir    : ${LOG_DIR}"
echo "Workers    : ${NUM_WORKERS}"
echo "Conc levels: ${CONC_LEVELS[*]}"
echo

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_aiperf() {
    local input_file="$1"
    local req_count="$2"
    local conc="$3"
    local artifact_dir="$4"

    if [[ -d "${artifact_dir}" ]]; then
        echo "  [skip] ${artifact_dir} already exists"
        return
    fi

    echo "  aiperf: $(basename "${input_file}") req=${req_count} conc=${conc} -> $(basename "${artifact_dir}")"
    aiperf profile \
        --model "${MODEL}" \
        --input-file "${input_file}" \
        --custom-dataset-type single_turn \
        --osl 1 \
        --request-count "${req_count}" \
        --concurrency "${conc}" \
        --artifact-dir "${artifact_dir}"
    sleep "${SLEEP_BETWEEN}"
}

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
run_warmup() {
    local warmup_file="${DATASET_DIR}/warmup_50req_1img_5pool_http.jsonl"
    echo "=== Warmup ==="
    rm -rf "${LOG_DIR}/warmup_http"
    run_aiperf "${warmup_file}" 50 "${WARMUP_CONC}" "${LOG_DIR}/warmup_http"
    echo
}

# ---------------------------------------------------------------------------
# Common inner loop: iterate over modes/pools for a given prefix
# ---------------------------------------------------------------------------
_run_pools() {
    local prefix="$1"   # e.g. "mm", "rr", "no_mm"
    for conc in "${CONC_LEVELS[@]}"; do
        local n=$(( conc * 100 ))
        local pool_90=$(( n / 10 ))
        local pool_50=$(( n * 6 / 10 ))
        for mode in http datauri; do
            echo "=== ${prefix} | ${mode} | conc=${conc} | req=${n} ==="
            for pool in "${pool_90}" "${pool_50}" "${n}"; do
                local input="${DATASET_DIR}/${n}req_1img_${pool}pool_${mode}.jsonl"
                local out="${LOG_DIR}/${prefix}_${NUM_WORKERS}w_${n}req_1img_${pool}pool_${mode}_conc${conc}"
                run_aiperf "${input}" "${n}" "${conc}" "${out}"
            done
            echo
        done
    done
}

# ---------------------------------------------------------------------------
# Per-router sweeps
# ---------------------------------------------------------------------------
run_mm()    { _run_pools "mm"; }
run_rr()    { _run_pools "rr"; }
run_no_mm() { _run_pools "no_mm"; }
run_rust()  { _run_pools "rust"; }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run_warmup

case "${ROUTER}" in
    mm)    run_mm ;;
    rr)    run_rr ;;
    no-mm) run_no_mm ;;
    rust)  run_rust ;;
    all)   run_mm; run_rr; run_no_mm; run_rust ;;
    *)     echo "Unknown --router value: ${ROUTER} (use mm | rr | no-mm | rust | all)"; exit 1 ;;
esac

echo "=== Sweep complete. Logs in ${LOG_DIR} ==="
