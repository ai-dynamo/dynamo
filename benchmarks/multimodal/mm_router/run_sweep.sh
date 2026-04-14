#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Concurrency sweep for MM routing benchmarks.
#
# Router types:
#   frontend : NEW — Frontend vLLM processor + KvRouter + NIXL (no MM Router Worker)
#   mm       : OLD — Frontend (round-robin) -> MM Router Worker -> vLLM backend
#   rr       : BASELINE — Frontend (round-robin) -> vLLM backend (no MM routing)
#
# Modes: http, datauri
# Concurrency: 1, 4, 8, 16, 32, 64
# Pools: 10pool (~90% reuse), 60pool (~50% reuse), Npool (0% reuse)
#
# Usage:
#   bash run_sweep.sh --router frontend --dataset-dir ./datasets --log-dir ./logs --conc 1
#   bash run_sweep.sh --router mm       --dataset-dir ./datasets --log-dir ./logs --conc 1
#   bash run_sweep.sh --router rr       --dataset-dir ./datasets --log-dir ./logs --conc 1
#   bash run_sweep.sh --router all      --dataset-dir ./datasets --log-dir ./logs

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
URL="${AIPERF_URL:-http://127.0.0.1:8000}"
ROUTER="${ROUTER:-all}"
DATASET_DIR="${DATASET_DIR:-$(dirname "$0")/datasets}"
LOG_DIR="${LOG_DIR:-$(dirname "$0")/logs}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_CONC="${WARMUP_CONC:-1}"

CONC_LEVELS=(1 4 8 16 32 64)

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

echo "=== MM Routing Benchmark Sweep ==="
echo "Model      : ${MODEL}"
echo "Router     : ${ROUTER}  (frontend | mm | rr | all)"
echo "Dataset dir: ${DATASET_DIR}"
echo "Log dir    : ${LOG_DIR}"
echo "Workers    : ${NUM_WORKERS}"
echo "Conc levels: ${CONC_LEVELS[*]}"
echo

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
HTTP_PORT="${HTTP_PORT:-8000}"
VLLM_SYSTEM_PORT_BASE="${VLLM_SYSTEM_PORT_BASE:-18079}"

scrape_metrics() {
    # Scrape frontend + all backend workers, save to a single file
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

# Extract a single worker's section from the metrics file (between its header
# and the next worker header or EOF).
_worker_section() {
    local file="$1" worker_idx="$2"
    # Extract lines between "Backend worker N" header and the next worker header (or EOF).
    awk -v idx="${worker_idx}" '
        /^# === Backend worker / {
            if (found) exit          # hit next worker -> stop
            if ($5 == idx) found=1   # $5 is the worker index in "# === Backend worker N metrics ..."
            next
        }
        found { print }
    ' "${file}"
}

diff_worker_requests() {
    # Compare before/after metrics to show per-worker request deltas
    local before="$1"
    local after="$2"
    local summary_file="$3"
    {
        echo "=== Per-worker request counts ==="
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local port=$(( VLLM_SYSTEM_PORT_BASE + i * 2 ))
            local before_count=$(_worker_section "${before}" "${i}" | grep 'dynamo_component_requests_total.*generate' | head -1 | awk '{print $NF}')
            local after_count=$(_worker_section "${after}" "${i}" | grep 'dynamo_component_requests_total.*generate' | head -1 | awk '{print $NF}')
            before_count=${before_count:-0}
            after_count=${after_count:-0}
            local delta=$(python3 -c "print(int(float(${after_count})-float(${before_count})))" 2>/dev/null || echo "?")
            echo "  Worker ${i} (port ${port}): ${delta} requests (${before_count} -> ${after_count})"
        done

        echo ""
        echo "=== Frontend KV hit rate histogram ==="
        grep 'dynamo_component_router_kv_hit_rate' "${after}" | tail -5

        echo ""
        echo "=== Frontend stage durations ==="
        for stage in route transport_roundtrip; do
            local count=$(grep "dynamo_frontend_stage_duration_seconds_count{stage=\"${stage}\"}" "${after}" | head -1 | awk '{print $NF}')
            local sum=$(grep "dynamo_frontend_stage_duration_seconds_sum{stage=\"${stage}\"}" "${after}" | head -1 | awk '{print $NF}')
            if [[ -n "${count}" && -n "${sum}" && "${count}" != "0" ]]; then
                local avg=$(python3 -c "print(f'{float(${sum})/float(${count})*1000:.2f}')" 2>/dev/null || echo "?")
                echo "  ${stage}: count=${count} avg=${avg}ms"
            fi
        done

        echo ""
        echo "=== Per-worker TTFT (last observed) ==="
        grep 'dynamo_frontend_worker_last_time_to_first_token_seconds' "${after}" | while read -r line; do
            echo "  ${line}"
        done

        echo ""
        echo "=== Per-worker prefix cache hit rate ==="
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local queries=$(_worker_section "${after}" "${i}" | grep '^vllm:prefix_cache_queries_total ' | head -1 | awk '{print $NF}')
            local hits=$(_worker_section "${after}" "${i}" | grep '^vllm:prefix_cache_hits_total ' | head -1 | awk '{print $NF}')
            local b_queries=$(_worker_section "${before}" "${i}" | grep '^vllm:prefix_cache_queries_total ' | head -1 | awk '{print $NF}')
            local b_hits=$(_worker_section "${before}" "${i}" | grep '^vllm:prefix_cache_hits_total ' | head -1 | awk '{print $NF}')
            queries=${queries:-0}; hits=${hits:-0}; b_queries=${b_queries:-0}; b_hits=${b_hits:-0}
            local dq=$(python3 -c "print(float(${queries})-float(${b_queries}))" 2>/dev/null || echo "0")
            local dh=$(python3 -c "print(float(${hits})-float(${b_hits}))" 2>/dev/null || echo "0")
            local rate=$(python3 -c "q=${dq}; h=${dh}; print(f'{h/q*100:.1f}%' if q>0 else 'n/a')" 2>/dev/null || echo "?")
            echo "  Worker ${i}: ${rate} (${dh}/${dq} tokens)"
        done

        echo ""
        echo "=== Per-worker MM cache (hit/query) ==="
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local queries=$(_worker_section "${after}" "${i}" | grep '^vllm:mm_cache_queries_total ' | head -1 | awk '{print $NF}')
            local hits=$(_worker_section "${after}" "${i}" | grep '^vllm:mm_cache_hits_total ' | head -1 | awk '{print $NF}')
            local b_queries=$(_worker_section "${before}" "${i}" | grep '^vllm:mm_cache_queries_total ' | head -1 | awk '{print $NF}')
            local b_hits=$(_worker_section "${before}" "${i}" | grep '^vllm:mm_cache_hits_total ' | head -1 | awk '{print $NF}')
            queries=${queries:-0}; hits=${hits:-0}; b_queries=${b_queries:-0}; b_hits=${b_hits:-0}
            local dq=$(python3 -c "print(float(${queries})-float(${b_queries}))" 2>/dev/null || echo "0")
            local dh=$(python3 -c "print(float(${hits})-float(${b_hits}))" 2>/dev/null || echo "0")
            local rate=$(python3 -c "q=${dq}; h=${dh}; print(f'{h/q*100:.1f}%' if q>0 else 'n/a')" 2>/dev/null || echo "?")
            echo "  Worker ${i}: ${rate} (${dh}/${dq} items)"
        done

        echo ""
        echo "=== Per-worker GPU KV cache usage ==="
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local usage=$(_worker_section "${after}" "${i}" | grep '^vllm:kv_cache_usage_perc ' | head -1 | awk '{print $NF}')
            usage=${usage:-0}
            local pct=$(python3 -c "print(f'{float(${usage})*100:.1f}%')" 2>/dev/null || echo "?")
            echo "  Worker ${i}: ${pct}"
        done

        echo ""
        echo "=== Network transit time (frontend->backend, per worker) ==="
        for i in $(seq 1 "${NUM_WORKERS}"); do
            local transit_count=$(_worker_section "${after}" "${i}" | grep 'dynamo_work_handler_network_transit_seconds_count.*generate' | head -1 | awk '{print $NF}')
            local transit_sum=$(_worker_section "${after}" "${i}" | grep 'dynamo_work_handler_network_transit_seconds_sum.*generate' | head -1 | awk '{print $NF}')
            if [[ -n "${transit_count}" && -n "${transit_sum}" && "${transit_count}" != "0" ]]; then
                local avg=$(python3 -c "print(f'{float(${transit_sum})/float(${transit_count})*1000:.2f}')" 2>/dev/null || echo "?")
                echo "  Worker ${i}: avg=${avg}ms (${transit_count} requests)"
            else
                echo "  Worker ${i}: n/a"
            fi
        done
    } > "${summary_file}"
    cat "${summary_file}"
}

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

    # Scrape metrics before
    local metrics_dir="${artifact_dir}/metrics"
    mkdir -p "${metrics_dir}"
    scrape_metrics "${metrics_dir}/before.txt" || true

    # Build --url flags (supports multiple URLs for load balancing)
    local url_flags=""
    for u in ${URL}; do
        url_flags="${url_flags} --url ${u}"
    done

    aiperf profile \
        --model "${MODEL}" \
        ${url_flags} \
        --input-file "${input_file}" \
        --custom-dataset-type single_turn \
        --osl 1 \
        --request-count "${req_count}" \
        --concurrency "${conc}" \
        --artifact-dir "${artifact_dir}"

    # Scrape metrics after and compute summary.
    # Run in a subshell so grep/awk failures in metrics parsing
    # don't kill the script under set -e.
    scrape_metrics "${metrics_dir}/after.txt" || true
    echo "  --- Metrics summary ---"
    ( diff_worker_requests "${metrics_dir}/before.txt" "${metrics_dir}/after.txt" "${metrics_dir}/summary.txt" ) || echo "  [warn] metrics summary had errors"
    echo "  -----------------------"

    sleep "${SLEEP_BETWEEN}"
}

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
run_warmup() {
    local warmup_file="${DATASET_DIR}/warmup_50req_1img_5pool_http.jsonl"
    if [[ ! -f "${warmup_file}" ]]; then
        echo "WARNING: warmup file not found: ${warmup_file}"
        echo "  Run generate_datasets.sh first."
        return
    fi
    echo "=== Warmup ==="
    rm -rf "${LOG_DIR}/warmup_http"
    run_aiperf "${warmup_file}" 50 "${WARMUP_CONC}" "${LOG_DIR}/warmup_http"
    echo
}

# ---------------------------------------------------------------------------
# Common inner loop: iterate over modes/pools for a given prefix
# ---------------------------------------------------------------------------
_run_pools() {
    local prefix="$1"
    for conc in "${CONC_LEVELS[@]}"; do
        local n=$(( conc * 100 ))
        local pool_90=$(( n / 10 ))
        local pool_50=$(( n * 6 / 10 ))
        for mode in http datauri; do
            echo "=== ${prefix} | ${mode} | conc=${conc} | req=${n} ==="
            for pool in "${pool_90}" "${pool_50}" "${n}"; do
                local input="${DATASET_DIR}/${n}req_1img_${pool}pool_${mode}.jsonl"
                if [[ ! -f "${input}" ]]; then
                    echo "  [skip] ${input} not found"
                    continue
                fi
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
run_frontend()      { _run_pools "frontend"; }
run_frontend_pool() { _run_pools "frontend_pool"; }
run_mm()            { _run_pools "mm"; }
run_rr()            { _run_pools "rr"; }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
run_warmup

case "${ROUTER}" in
    frontend)      run_frontend ;;
    frontend_pool) run_frontend_pool ;;
    mm)            run_mm ;;
    rr)            run_rr ;;
    all)           run_frontend; run_frontend_pool; run_mm; run_rr ;;
    *)             echo "Unknown --router value: ${ROUTER} (use frontend | frontend_pool | mm | rr | all)"; exit 1 ;;
esac

echo "=== Sweep complete. Logs in ${LOG_DIR} ==="
