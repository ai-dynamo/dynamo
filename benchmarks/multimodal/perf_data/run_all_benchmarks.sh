#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MM Router A/B benchmark: sweeps b64 + localhost datasets at 100 requests.
#
# Usage:
#   DYNAMO_ROOT=/path/to/dynamo PERF_DIR=/path/to/this/dir bash run_all_benchmarks.sh
#
# Defaults:
#   DYNAMO_ROOT = parent of benchmarks/ (auto-detected)
#   PERF_DIR    = directory containing this script (auto-detected)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PERF_DIR="${PERF_DIR:-${SCRIPT_DIR}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${PERF_DIR}/results_${TIMESTAMP}"
LOG_FILE="${PERF_DIR}/benchmark_${TIMESTAMP}.log"

DATASETS=(
    "100req_1img_90pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_50pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_0pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_90pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_50pct_reuse_b64_512_fixprompt.jsonl"
    "100req_3img_0pct_reuse_b64_512_fixprompt.jsonl"
    "100req_1img_90pct_reuse_localhost_fixprompt.jsonl"
    "100req_1img_50pct_reuse_localhost_fixprompt.jsonl"
    "100req_1img_0pct_reuse_localhost_fixprompt.jsonl"
    "100req_3img_90pct_reuse_localhost_fixprompt.jsonl"
    "100req_3img_50pct_reuse_localhost_fixprompt.jsonl"
    "100req_3img_0pct_reuse_localhost_fixprompt.jsonl"
)

cd "${DYNAMO_ROOT}"

echo "=== Benchmark started at $(date) ===" | tee -a "${LOG_FILE}"
echo "DYNAMO_ROOT: ${DYNAMO_ROOT}" | tee -a "${LOG_FILE}"
echo "PERF_DIR:    ${PERF_DIR}" | tee -a "${LOG_FILE}"
echo "Results:     ${RESULT_ROOT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for ds in "${DATASETS[@]}"; do
    ds_label="${ds%.jsonl}"
    ds_result="${RESULT_ROOT}/${ds_label}"

    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "  Dataset: ${ds}" | tee -a "${LOG_FILE}"
    echo "  Started: $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"

    env \
        MODEL="Qwen/Qwen3-VL-2B-Instruct" \
        DATASET="${PERF_DIR}/${ds}" \
        REQUEST_COUNT=100 \
        WARMUP_REQUEST_COUNT=10 \
        CONCURRENCIES="1" \
        OSL=1 \
        HTTP_PORT=8000 \
        RESULT_ROOT="${ds_result}" \
        RUN_BASELINE=1 \
        RUN_MM_ROUTER=1 \
        CLEAN_BEFORE_START=1 \
        RESTART_STACK_EACH_CONCURRENCY=1 \
        READINESS_TIMEOUT_SECS=600 \
        PROBE_TIMEOUT_SECS=180 \
        bash examples/backends/vllm/mm_router_worker/run_aiperf_ab.sh 2>&1 | tee -a "${LOG_FILE}"

    echo "" | tee -a "${LOG_FILE}"
    echo "  Finished: $(date)" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
done

echo "=== All benchmarks completed at $(date) ===" | tee -a "${LOG_FILE}"
echo "Results directory: ${RESULT_ROOT}" | tee -a "${LOG_FILE}"
