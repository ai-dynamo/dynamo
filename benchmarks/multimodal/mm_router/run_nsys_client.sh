#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Client script for nsys profiling sessions.
#
# Run this from a SECOND terminal after starting the server under nsys.
# Sends warmup requests, then a small measured batch with metrics capture.
#
# Usage:
#   bash run_nsys_client.sh                          # defaults: conc=1, 20 warmup, 30 measured
#   bash run_nsys_client.sh --conc 8                 # conc=8
#   bash run_nsys_client.sh --conc 1 --label mm      # custom label for output dir
#   bash run_nsys_client.sh --mode http              # http URLs instead of datauri
#   bash run_nsys_client.sh --pool 90                # ~90% reuse (default)
#   bash run_nsys_client.sh --pool 50                # ~50% reuse
#   bash run_nsys_client.sh --pool 0                 # 0% reuse
#   bash run_nsys_client.sh --warmup-only            # just warmup, no measured
#   bash run_nsys_client.sh --measured-only           # skip warmup (already warmed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
URL="${AIPERF_URL:-http://127.0.0.1:8000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATASET_DIR="${DATASET_DIR:-${SCRIPT_DIR}/datasets}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/nsys_client_results}"

CONC=1
MODE="datauri"
POOL="90"         # 90, 50, or 0
LABEL=""
WARMUP_COUNT=20
MEASURED_COUNT=30
SLEEP_BETWEEN=3

DO_WARMUP=1
DO_MEASURED=1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conc)          CONC="$2";            shift 2 ;;
        --mode)          MODE="$2";            shift 2 ;;
        --pool)          POOL="$2";            shift 2 ;;
        --label)         LABEL="$2";           shift 2 ;;
        --model)         MODEL="$2";           shift 2 ;;
        --url)           URL="$2";             shift 2 ;;
        --workers)       NUM_WORKERS="$2";     shift 2 ;;
        --dataset-dir)   DATASET_DIR="$2";     shift 2 ;;
        --out-dir)       OUT_DIR="$2";         shift 2 ;;
        --warmup-count)  WARMUP_COUNT="$2";    shift 2 ;;
        --measured-count) MEASURED_COUNT="$2";  shift 2 ;;
        --warmup-only)   DO_MEASURED=0;        shift ;;
        --measured-only) DO_WARMUP=0;          shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Metrics (same as run_sweep.sh)
# ---------------------------------------------------------------------------
HTTP_PORT="${HTTP_PORT:-8000}"
VLLM_SYSTEM_PORT_BASE="${VLLM_SYSTEM_PORT_BASE:-18079}"

scrape_metrics() {
    local out_file="$1"
    {
        echo "# === Frontend metrics (port ${HTTP_PORT}) ==="
        curl -s "http://127.0.0.1:${HTTP_PORT}/metrics" 2>/dev/null || echo "# frontend scrape failed"
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local port=$(( VLLM_SYSTEM_PORT_BASE + i * 2 ))
            echo ""
            echo "# === Backend worker ${i} metrics (port ${port}) ==="
            curl -s "http://127.0.0.1:${port}/metrics" 2>/dev/null || echo "# worker ${i} scrape failed"
        done
    } > "${out_file}"
}

_worker_section() {
    local file="$1" worker_idx="$2"
    awk -v idx="${worker_idx}" '
        /^# === Backend worker / {
            if (found) exit
            if ($5 == idx) found=1
            next
        }
        found { print }
    ' "${file}"
}

print_metrics_summary() {
    local before="$1" after="$2"

    echo ""
    echo "  === Per-worker request counts ==="
    for i in $(seq 1 "${NUM_WORKERS}"); do
        local port=$(( VLLM_SYSTEM_PORT_BASE + i * 2 ))
        local bc=$(_worker_section "${before}" "${i}" | grep 'dynamo_component_requests_total.*generate' | head -1 | awk '{print $NF}')
        local ac=$(_worker_section "${after}" "${i}" | grep 'dynamo_component_requests_total.*generate' | head -1 | awk '{print $NF}')
        bc=${bc:-0}; ac=${ac:-0}
        local delta=$(python3 -c "print(int(float(${ac})-float(${bc})))" 2>/dev/null || echo "?")
        echo "    Worker ${i} (port ${port}): ${delta} requests"
    done

    echo ""
    echo "  === Per-worker prefix cache hit rate ==="
    for i in $(seq 1 "${NUM_WORKERS}"); do
        local queries=$(_worker_section "${after}" "${i}" | grep '^vllm:prefix_cache_queries_total' | head -1 | awk '{print $NF}')
        local hits=$(_worker_section "${after}" "${i}" | grep '^vllm:prefix_cache_hits_total' | head -1 | awk '{print $NF}')
        local b_queries=$(_worker_section "${before}" "${i}" | grep '^vllm:prefix_cache_queries_total' | head -1 | awk '{print $NF}')
        local b_hits=$(_worker_section "${before}" "${i}" | grep '^vllm:prefix_cache_hits_total' | head -1 | awk '{print $NF}')
        queries=${queries:-0}; hits=${hits:-0}; b_queries=${b_queries:-0}; b_hits=${b_hits:-0}
        local dq=$(python3 -c "print(float(${queries})-float(${b_queries}))" 2>/dev/null || echo "0")
        local dh=$(python3 -c "print(float(${hits})-float(${b_hits}))" 2>/dev/null || echo "0")
        local rate=$(python3 -c "q=${dq}; h=${dh}; print(f'{h/q*100:.1f}%' if q>0 else 'n/a')" 2>/dev/null || echo "?")
        echo "    Worker ${i}: ${rate} (${dh}/${dq} tokens)"
    done

    echo ""
    echo "  === Per-worker MM cache (hit/query) ==="
    for i in $(seq 1 "${NUM_WORKERS}"); do
        local queries=$(_worker_section "${after}" "${i}" | grep '^vllm:mm_cache_queries_total' | head -1 | awk '{print $NF}')
        local hits=$(_worker_section "${after}" "${i}" | grep '^vllm:mm_cache_hits_total' | head -1 | awk '{print $NF}')
        local b_queries=$(_worker_section "${before}" "${i}" | grep '^vllm:mm_cache_queries_total' | head -1 | awk '{print $NF}')
        local b_hits=$(_worker_section "${before}" "${i}" | grep '^vllm:mm_cache_hits_total' | head -1 | awk '{print $NF}')
        queries=${queries:-0}; hits=${hits:-0}; b_queries=${b_queries:-0}; b_hits=${b_hits:-0}
        local dq=$(python3 -c "print(float(${queries})-float(${b_queries}))" 2>/dev/null || echo "0")
        local dh=$(python3 -c "print(float(${hits})-float(${b_hits}))" 2>/dev/null || echo "0")
        local rate=$(python3 -c "q=${dq}; h=${dh}; print(f'{h/q*100:.1f}%' if q>0 else 'n/a')" 2>/dev/null || echo "?")
        echo "    Worker ${i}: ${rate} (${dh}/${dq} items)"
    done

    echo ""
    echo "  === Network transit time (frontend->backend) ==="
    for i in $(seq 1 "${NUM_WORKERS}"); do
        local tc=$(_worker_section "${after}" "${i}" | grep 'dynamo_work_handler_network_transit_seconds_count.*generate' | head -1 | awk '{print $NF}')
        local ts=$(_worker_section "${after}" "${i}" | grep 'dynamo_work_handler_network_transit_seconds_sum.*generate' | head -1 | awk '{print $NF}')
        if [[ -n "${tc}" && -n "${ts}" && "${tc}" != "0" ]]; then
            local avg=$(python3 -c "print(f'{float(${ts})/float(${tc})*1000:.2f}')" 2>/dev/null || echo "?")
            echo "    Worker ${i}: avg=${avg}ms (${tc} requests)"
        else
            echo "    Worker ${i}: n/a"
        fi
    done
}

# ---------------------------------------------------------------------------
# Resolve dataset file
# ---------------------------------------------------------------------------
# Dataset naming: {N}req_1img_{pool}pool_{mode}.jsonl
# N = conc * 100, pool depends on --pool flag
N=$(( CONC * 100 ))
case "${POOL}" in
    90) POOL_SIZE=$(( N / 10 )) ;;
    50) POOL_SIZE=$(( N * 6 / 10 )) ;;
    0)  POOL_SIZE="${N}" ;;
    *)  POOL_SIZE="${POOL}" ;;   # allow exact pool size
esac

INPUT_FILE="${DATASET_DIR}/${N}req_1img_${POOL_SIZE}pool_${MODE}.jsonl"
WARMUP_FILE="${DATASET_DIR}/warmup_50req_1img_5pool_http.jsonl"

# Build --url flags
URL_FLAGS=""
for u in ${URL}; do
    URL_FLAGS="${URL_FLAGS} --url ${u}"
done

# Label for output dirs
[[ -z "${LABEL}" ]] && LABEL="nsys_${MODE}_${POOL}pool"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "  nsys Client"
echo "  Model   : ${MODEL}"
echo "  URL     : ${URL}"
echo "  Workers : ${NUM_WORKERS}"
echo "  Conc    : ${CONC}"
echo "  Mode    : ${MODE}"
echo "  Pool    : ~${POOL}% reuse (pool_size=${POOL_SIZE})"
echo "  Dataset : ${INPUT_FILE}"
echo "  Warmup  : ${WARMUP_COUNT} requests"
echo "  Measured: ${MEASURED_COUNT} requests"
echo "  Output  : ${OUT_DIR}"
echo "============================================================"
echo ""

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "ERROR: Dataset not found: ${INPUT_FILE}"
    echo ""
    echo "Generate it with:"
    echo "  bash generate_datasets.sh ${DATASET_DIR} ${CONC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
if [[ "${DO_WARMUP}" -eq 1 ]]; then
    echo "=== Warmup: ${WARMUP_COUNT} requests at conc=${CONC} ==="

    warmup_dir="${OUT_DIR}/${LABEL}_conc${CONC}_warmup"
    rm -rf "${warmup_dir}"

    aiperf profile \
        --model "${MODEL}" \
        ${URL_FLAGS} \
        --input-file "${INPUT_FILE}" \
        --custom-dataset-type single_turn \
        --osl 1 \
        --request-count "${WARMUP_COUNT}" \
        --concurrency "${CONC}" \
        --artifact-dir "${warmup_dir}" 2>&1 | tail -5

    echo "Warmup done. Sleeping ${SLEEP_BETWEEN}s..."
    sleep "${SLEEP_BETWEEN}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Measured
# ---------------------------------------------------------------------------
if [[ "${DO_MEASURED}" -eq 1 ]]; then
    echo "=== Measured: ${MEASURED_COUNT} requests at conc=${CONC} ==="

    measured_dir="${OUT_DIR}/${LABEL}_conc${CONC}_measured"
    rm -rf "${measured_dir}"
    metrics_dir="${measured_dir}/metrics"
    mkdir -p "${metrics_dir}"

    # Scrape before
    scrape_metrics "${metrics_dir}/before.txt" || true

    aiperf profile \
        --model "${MODEL}" \
        ${URL_FLAGS} \
        --input-file "${INPUT_FILE}" \
        --custom-dataset-type single_turn \
        --osl 1 \
        --request-count "${MEASURED_COUNT}" \
        --concurrency "${CONC}" \
        --artifact-dir "${measured_dir}"

    # Scrape after + summary
    scrape_metrics "${metrics_dir}/after.txt" || true
    echo ""
    echo "=== Metrics Summary ==="
    ( print_metrics_summary "${metrics_dir}/before.txt" "${metrics_dir}/after.txt" ) || echo "  [warn] metrics summary had errors"
    echo ""
    echo "Results: ${measured_dir}"
fi

echo ""
echo "=== Done ==="
echo "You can now Ctrl+C the nsys server terminal to finalize the trace."
