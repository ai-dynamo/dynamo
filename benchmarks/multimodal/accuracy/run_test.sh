#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0"
EVAL_IMAGE="nvcr.io/nvidia/eval-factory/vlmevalkit:26.01"
CONTAINER_NAME="vllm-accuracy-server"

echo "============================================"
echo "  Accuracy Test: Qwen/Qwen3-VL-2B-Instruct"
echo "============================================"
echo ""

# Step 1: Pull containers
echo "=== Step 1: Pulling containers ==="
docker pull "${VLLM_IMAGE}"
docker pull "${EVAL_IMAGE}"
echo ""

# Step 2: Start model server
echo "=== Step 2: Starting model server ==="
"${SCRIPT_DIR}/serve_model.sh"
echo ""

# Step 3: Run benchmarks
echo "=== Step 3: Running accuracy benchmarks ==="
export RESULTS_DIR="${SCRIPT_DIR}/results"
"${SCRIPT_DIR}/run_accuracy_test.sh"
echo ""

# Step 4: Summary
echo "=== Step 4: Results Summary ==="
for BENCH_DIR in "${RESULTS_DIR}"/*/; do
    BENCH_NAME=$(basename "${BENCH_DIR}")
    echo ""
    echo "--- ${BENCH_NAME} ---"
    if [ -f "${BENCH_DIR}/results.yml" ]; then
        cat "${BENCH_DIR}/results.yml"
    else
        echo "  (no results.yml found)"
        ls -la "${BENCH_DIR}" 2>/dev/null || echo "  (empty)"
    fi
done

# Step 5: Cleanup server
echo ""
echo "=== Step 5: Cleaning up ==="
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
echo "Server container removed."

echo ""
echo "=== Done! Results are in: ${RESULTS_DIR} ==="
