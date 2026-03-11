#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Cluster benchmark script for Qwen3-VL-30B-A3B-Instruct-FP8 on 8x B200
#
# This script runs the same A/B comparison (baseline RR vs MM router) as
# run_all_benchmarks.sh, but scaled to 8 workers on cluster GPUs.
#
# Usage:
#   DYNAMO_ROOT=/path/to/dynamo PERF_DIR=/path/to/this/dir bash run_cluster_benchmarks.sh
#
# Prerequisites on the cluster node:
#   1. etcd + NATS running:  docker compose -f deploy/docker-compose.yml up -d
#   2. Dynamo installed (pip install -e . or container image)
#   3. aiperf installed (pip install aiperf)
#   4. Model accessible (HF cache or local path)
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PERF_DIR="${PERF_DIR:-${SCRIPT_DIR}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${PERF_DIR}/cluster_results_${TIMESTAMP}"
LOG_FILE="${PERF_DIR}/cluster_benchmark_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Model & cluster config
# ---------------------------------------------------------------------------
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
NUM_WORKERS=8

# Tuned for 30B-A3B MoE FP8 on B200 (192 GB per GPU, 1 worker per GPU)
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192

# ---------------------------------------------------------------------------
# Datasets — base64 encoded images (self-contained, no image server needed)
# ---------------------------------------------------------------------------
DATASETS=(
    "100req_1img_90pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_50pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_20pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_0pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_90pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_50pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_20pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_0pct_reuse_b64_512_fixprompt.jsonl"
)

# Concurrency levels to sweep (scale up with more workers)
CONCURRENCIES="1 2 4 8"

cd "${DYNAMO_ROOT}"

echo "=== Cluster Benchmark started at $(date) ===" | tee -a "${LOG_FILE}"
echo "Model:       ${MODEL}" | tee -a "${LOG_FILE}"
echo "Workers:     ${NUM_WORKERS}" | tee -a "${LOG_FILE}"
echo "DYNAMO_ROOT: ${DYNAMO_ROOT}" | tee -a "${LOG_FILE}"
echo "PERF_DIR:    ${PERF_DIR}" | tee -a "${LOG_FILE}"
echo "Results:     ${RESULT_ROOT}" | tee -a "${LOG_FILE}"
echo "Concurrency: ${CONCURRENCIES}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for ds in "${DATASETS[@]}"; do
    ds_path="${PERF_DIR}/${ds}"
    if [[ ! -f "${ds_path}" ]]; then
        echo "SKIP: dataset not found: ${ds_path}" | tee -a "${LOG_FILE}"
        continue
    fi

    ds_label="${ds%.jsonl}"
    ds_result="${RESULT_ROOT}/${ds_label}"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "  Dataset: ${ds}" | tee -a "${LOG_FILE}"
    echo "  Started: $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    env \
        MODEL="${MODEL}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
        DATASET="${ds_path}" \
        REQUEST_COUNT=100 \
        WARMUP_REQUEST_COUNT=10 \
        CONCURRENCIES="${CONCURRENCIES}" \
        OSL=1 \
        HTTP_PORT=8000 \
        RESULT_ROOT="${ds_result}" \
        RUN_BASELINE=1 \
        RUN_MM_ROUTER=1 \
        CLEAN_BEFORE_START=1 \
        RESTART_STACK_EACH_CONCURRENCY=1 \
        READINESS_TIMEOUT_SECS=900 \
        PROBE_TIMEOUT_SECS=300 \
        bash examples/backends/vllm/mm_router_worker/run_aiperf_ab.sh 2>&1 | tee -a "${LOG_FILE}"

    echo "" | tee -a "${LOG_FILE}"
    echo "  Finished: $(date)" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
done

echo "=== All benchmarks completed at $(date) ===" | tee -a "${LOG_FILE}"
echo "Results directory: ${RESULT_ROOT}" | tee -a "${LOG_FILE}"
