#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark sweep: multimodal embedding cache OFF (control) vs ON (test)
# for disagg_multimodal_e_pd.sh
#
# Usage:
#   ./benchmarks/multimodal/benchmark_sweep.sh [OPTIONS]
#
# Options:
#   --workflow <script>     Launch script path relative to repo root (repeatable)
#                           (default: examples/backends/vllm/launch/disagg_multimodal_e_pd.sh)
#   --model <model>         Model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
#   --concurrencies <list>  Comma-separated concurrency levels (default: 1,2,5,10)
#   --osl <tokens>          Output sequence length (default: 150)
#   --request-count <n>     Requests per concurrency level (default: 100)
#   --warmup <n>            Warmup requests (default: 5)
#   --cache-gb <gb>         Embedding cache capacity in GB for test (default: 2)
#   --input-file <path>     JSONL input file (required, repeatable for multiple files)
#   --output-dir <dir>      Results output directory (default: benchmarks/results/embedding_cache_sweep)
#   --port <port>           Frontend port to poll for readiness (default: 8000)
#   --timeout <sec>         Max seconds to wait for server readiness (default: 600)
#   --skip-control          Skip the control run (cache OFF)
#   --skip-test             Skip the test run (cache ON)
#   --skip-plots            Skip plot generation
#   -h, --help              Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Defaults
WORKFLOWS=()
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
CONCURRENCIES="1,2,4,8,16,32,64,128,256"
OSL=1
REQUEST_COUNT=1000
WARMUP_COUNT=5
CACHE_GB=10
INPUT_FILES=()
OUTPUT_DIR="benchmarks/results/multimodal_default"
PORT=8000
TIMEOUT=600
SKIP_CONTROL=false
SKIP_TEST=false
SKIP_PLOTS=false

# disable standalone encoder's cache
export ENABLE_ENCODER_CACHE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow)
            WORKFLOWS+=("$2")
            shift 2
            ;;
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --concurrencies)
            CONCURRENCIES=$2
            shift 2
            ;;
        --osl)
            OSL=$2
            shift 2
            ;;
        --request-count)
            REQUEST_COUNT=$2
            shift 2
            ;;
        --warmup)
            WARMUP_COUNT=$2
            shift 2
            ;;
        --cache-gb)
            CACHE_GB=$2
            shift 2
            ;;
        --input-file)
            INPUT_FILES+=("$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        --timeout)
            TIMEOUT=$2
            shift 2
            ;;
        --skip-control)
            SKIP_CONTROL=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        -h|--help)
            # Print usage from the header comment
            sed -n '6,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

IFS=',' read -ra CONCURRENCY_LIST <<< "$CONCURRENCIES"

if [[ ${#WORKFLOWS[@]} -eq 0 ]]; then
    WORKFLOWS=("examples/backends/vllm/launch/disagg_multimodal_e_pd.sh")
fi

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: At least one --input-file is required."
    echo "Use --help for usage information."
    exit 1
fi

# Validate all workflow scripts upfront.
for wf in "${WORKFLOWS[@]}"; do
    if [[ ! -x "$REPO_ROOT/$wf" ]]; then
        echo "ERROR: Workflow script not found or not executable: $REPO_ROOT/$wf"
        exit 1
    fi
done

# Derive a workflow tag from its path.
workflow_tag() {
    local wf=$1
    if [[ "$wf" == *vllm* && "$wf" == *e_pd* ]]; then
        echo "vllm_e_pd"
    elif [[ "$wf" == *trtllm* && "$wf" == *e_pd* ]]; then
        echo "trtllm_e_pd"
    elif [[ "$wf" == *vllm_serve* ]]; then
        echo "vllm_serve"
    else
        basename "$wf" .sh | tr ' ' '_'
    fi
}

BASE_OUTPUT_DIR="$OUTPUT_DIR"

echo "=================================================="
echo "Multimodal Benchmark Sweep"
echo "=================================================="
echo "Workflows:     ${WORKFLOWS[*]}"
echo "Model:         $MODEL_NAME"
echo "Input files:   ${INPUT_FILES[*]}"
echo "Concurrencies: ${CONCURRENCY_LIST[*]}"
echo "OSL:           $OSL"
echo "Requests:      $REQUEST_COUNT per concurrency"
echo "Cache (test):  ${CACHE_GB} GB"
echo "Output:        $BASE_OUTPUT_DIR"
echo "=================================================="

# ---------------------------------------------------------------------------
# Helper: wait for the server to become ready
# ---------------------------------------------------------------------------
wait_for_server() {
    local url="http://localhost:${PORT}/v1/models"
    local deadline=$((SECONDS + TIMEOUT))
    echo "Waiting for server at $url to list model '$MODEL_NAME' (timeout: ${TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if curl -sf "$url" 2>/dev/null | grep -q "$MODEL_NAME"; then
            echo "Server is ready (model registered)."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server did not become ready within ${TIMEOUT}s"
    return 1
}

# ---------------------------------------------------------------------------
# Helper: launch the serving stack and return its PID
# ---------------------------------------------------------------------------
SERVER_PID=""

start_server() {
    local extra_args=("${CUR_DEFAULT_SERVER_ARGS[@]}" "$@")
    echo "Launching $CUR_WORKFLOW ${extra_args[*]:-}..."
    setsid bash "$CUR_LAUNCH_SCRIPT" \
        --model "$MODEL_NAME" \
        "${extra_args[@]}" &
    SERVER_PID=$!
    wait_for_server
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping server (PID $SERVER_PID)..."
        # setsid gave the server its own process group (PGID == SERVER_PID),
        # so this kills only the server tree, not the sweep script.
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 5
    fi
}

cleanup() {
    echo "Cleaning up..."
    stop_server
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: run aiperf concurrency sweep
# ---------------------------------------------------------------------------
run_sweep() {
    local label=$1
    local input_file=$2
    local result_dir="$OUTPUT_DIR/$label"
    mkdir -p "$result_dir"

    for c in "${CONCURRENCY_LIST[@]}"; do
        local c_dir="$result_dir/c${c}"
        mkdir -p "$c_dir"
        echo ""
        echo "--- [$label] concurrency=$c ---"
        aiperf profile \
            -m "$MODEL_NAME" \
            -u "http://localhost:${PORT}" \
            --concurrency "$c" \
            --request-count "$REQUEST_COUNT" \
            --warmup-request-count "$WARMUP_COUNT" \
            --input-file "$input_file" \
            --custom-dataset-type single_turn \
            --extra-inputs "max_tokens:${OSL}" \
            --extra-inputs "min_tokens:${OSL}" \
            --extra-inputs "ignore_eos:true" \
            --extra-inputs "stream:true" \
            --artifact-dir "$c_dir" \
            --ui none \
            --no-server-metrics
    done
    echo ""
    echo "[$label] sweep complete. Results in $result_dir"
}

# Derive a short tag from a JSONL filename for use in directory names.
input_file_tag() {
    basename "$1" .jsonl | tr ' ' '_'
}

# ---------------------------------------------------------------------------
# Main loop: iterate over each workflow, then each input file
# ---------------------------------------------------------------------------
for CUR_WORKFLOW in "${WORKFLOWS[@]}"; do
    CUR_WORKFLOW_TAG="$(workflow_tag "$CUR_WORKFLOW")"
    CUR_LAUNCH_SCRIPT="$REPO_ROOT/$CUR_WORKFLOW"

    if [[ "$CUR_WORKFLOW" == *vllm* ]]; then
        CUR_DEFAULT_SERVER_ARGS=("--no-enable-prefix-caching")
    else
        CUR_DEFAULT_SERVER_ARGS=("--override-engine-args" '{"kv_cache_config": {"enable_block_reuse": false}}')
    fi

    echo ""
    echo "==================================================================="
    echo "# Workflow: $CUR_WORKFLOW"
    echo "# Tag:      $CUR_WORKFLOW_TAG"
    echo "==================================================================="

    for INPUT_FILE in "${INPUT_FILES[@]}"; do
        FILE_TAG="$(input_file_tag "$INPUT_FILE")"
        FILE_OUTPUT_DIR="$BASE_OUTPUT_DIR/$CUR_WORKFLOW_TAG/$FILE_TAG"

        echo ""
        echo "###################################################################"
        echo "# Input: $INPUT_FILE"
        echo "# Tag:   $FILE_TAG"
        echo "###################################################################"

        SAVE_OUTPUT_DIR="$OUTPUT_DIR"
        OUTPUT_DIR="$FILE_OUTPUT_DIR"

        # --- Control: embedding cache OFF ---
        if [[ "$SKIP_CONTROL" != "true" ]]; then
            echo ""
            echo "========== [$CUR_WORKFLOW_TAG/$FILE_TAG] CONTROL: embedding cache OFF =========="
            start_server --multimodal-embedding-cache-capacity-gb 0
            run_sweep "cache-off" "$INPUT_FILE"
            stop_server
        fi

        # --- Test: embedding cache ON ---
        if [[ "$SKIP_TEST" != "true" ]]; then
            echo ""
            echo "========== [$CUR_WORKFLOW_TAG/$FILE_TAG] TEST: embedding cache ON (${CACHE_GB} GB) =========="
            start_server --multimodal-embedding-cache-capacity-gb "$CACHE_GB"
            run_sweep "cache-on" "$INPUT_FILE"
            stop_server
        fi

        # --- Plots for this input file ---
        if [[ "$SKIP_PLOTS" != "true" ]]; then
            echo ""
            echo "========== [$CUR_WORKFLOW_TAG/$FILE_TAG] Generating comparison plots =========="
            python -m benchmarks.utils.plot \
                --data-dir "$FILE_OUTPUT_DIR" \
                --benchmark-name cache-off \
                --benchmark-name cache-on
            echo "Plots saved to: $FILE_OUTPUT_DIR/plots/"
        fi

        OUTPUT_DIR="$SAVE_OUTPUT_DIR"
    done
done

echo ""
echo "=================================================="
echo "Sweep complete!"
echo "  Results: $BASE_OUTPUT_DIR"
for CUR_WORKFLOW in "${WORKFLOWS[@]}"; do
    CUR_WORKFLOW_TAG="$(workflow_tag "$CUR_WORKFLOW")"
    echo "  [$CUR_WORKFLOW_TAG]:"
    for INPUT_FILE in "${INPUT_FILES[@]}"; do
        FILE_TAG="$(input_file_tag "$INPUT_FILE")"
        echo "    [$FILE_TAG]: $BASE_OUTPUT_DIR/$CUR_WORKFLOW_TAG/$FILE_TAG/"
        if [[ "$SKIP_PLOTS" != "true" ]]; then
            echo "      Plots: $BASE_OUTPUT_DIR/$CUR_WORKFLOW_TAG/$FILE_TAG/plots/"
        fi
    done
done
echo "=================================================="
