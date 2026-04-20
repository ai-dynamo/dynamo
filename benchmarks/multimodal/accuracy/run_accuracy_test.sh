#!/bin/bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
MODEL_URL="${MODEL_URL:-http://localhost:8000/v1/chat/completions}"
EVAL_IMAGE="nvcr.io/nvidia/eval-factory/vlmevalkit:26.01"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-}"  # empty = full dataset, set to e.g. 10 for smoke test
BENCHMARKS="${BENCHMARKS:-chartqa ocrbench}"

mkdir -p "${RESULTS_DIR}"

echo "=== Accuracy Test Configuration ==="
echo "Model:      ${MODEL_ID}"
echo "Endpoint:   ${MODEL_URL}"
echo "Eval Image: ${EVAL_IMAGE}"
echo "Results:    ${RESULTS_DIR}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Limit:      ${LIMIT_SAMPLES:-full dataset}"
echo "======================================"

# Verify the model endpoint is reachable
SERVER_BASE="${MODEL_URL%/v1/chat/completions}"
if ! curl -sf "${SERVER_BASE}/health" > /dev/null 2>&1; then
    echo "ERROR: Model endpoint not reachable at ${SERVER_BASE}/health"
    echo "Start the server first with ./serve_model.sh"
    exit 1
fi

for BENCHMARK in ${BENCHMARKS}; do
    echo ""
    echo "=== Running benchmark: ${BENCHMARK} ==="
    BENCH_RESULTS="${RESULTS_DIR}/${BENCHMARK}"
    mkdir -p "${BENCH_RESULTS}"

    LIMIT_ARG=""
    if [ -n "${LIMIT_SAMPLES}" ]; then
        LIMIT_ARG="--overrides config.params.limit_samples=${LIMIT_SAMPLES}"
    fi

    docker run --rm \
        --network host \
        -v "${BENCH_RESULTS}":/workspace/results \
        -e DUMMY_API_KEY="no-key-needed" \
        "${EVAL_IMAGE}" \
        nemo-evaluator run_eval \
            --eval_type "${BENCHMARK}" \
            --model_id "${MODEL_ID}" \
            --model_url "${MODEL_URL}" \
            --model_type vlm \
            --api_key_name DUMMY_API_KEY \
            --output_dir /workspace/results \
            ${LIMIT_ARG}

    echo "=== Benchmark ${BENCHMARK} complete ==="
    echo "Results saved to: ${BENCH_RESULTS}"

    # Print results summary if results.yml exists
    if [ -f "${BENCH_RESULTS}/results.yml" ]; then
        echo "--- Results Summary ---"
        cat "${BENCH_RESULTS}/results.yml"
        echo "--- End Summary ---"
    fi
done

echo ""
echo "=== All benchmarks complete ==="
echo "Results directory: ${RESULTS_DIR}"
ls -la "${RESULTS_DIR}"
