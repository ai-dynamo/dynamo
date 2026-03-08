#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dry-run: cycle through every deployment config (agg, epd-1e/2e/4e/6e),
# send a handful of requests, and report pass/fail for each.
#
# Usage:
#   bash benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/dry_run.sh
#
# Requires:
#   - 4x B200 GPUs visible
#   - One JSONL input file (ISL=400 by default, override with DRY_RUN_INPUT_FILE)
#   - aiperf installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=${MODEL:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
PORT=${PORT:-8000}
TIMEOUT=${TIMEOUT:-900}
OSL=${OSL:-50}
REQUEST_COUNT=${REQUEST_COUNT:-3}
WARMUP_COUNT=${WARMUP_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-1}
INPUT_FILE=${DRY_RUN_INPUT_FILE:-"benchmarks/multimodal/jsonl/1000req_1img_200pool_400word_http.jsonl"}

CONFIGS=(
    "agg:$SCRIPT_DIR/launch_agg.sh"
    "epd-1e:$SCRIPT_DIR/launch_epd_1e.sh"
    "epd-2e:$SCRIPT_DIR/launch_epd_2e.sh"
    "epd-4e:$SCRIPT_DIR/launch_epd_4e.sh"
    "epd-6e:$SCRIPT_DIR/launch_epd_6e.sh"
)

SERVER_PID=""

cleanup_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  Stopping server (PGID $SERVER_PID)..."
        kill -- -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        sleep 5
    fi
    SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

wait_for_model() {
    local url="http://localhost:${PORT}/v1/models"
    local deadline=$((SECONDS + TIMEOUT))

    echo "  Waiting for model at $url (timeout: ${TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if curl -sf "$url" 2>/dev/null | grep -q "$MODEL"; then
            echo "  Model ready."
            return 0
        fi
        sleep 5
    done
    echo "  TIMEOUT: model did not become ready within ${TIMEOUT}s"
    return 1
}

run_aiperf() {
    aiperf profile \
        -m "$MODEL" \
        -u "http://localhost:${PORT}" \
        --concurrency "$CONCURRENCY" \
        --request-count "$REQUEST_COUNT" \
        --warmup-request-count "$WARMUP_COUNT" \
        --input-file "$INPUT_FILE" \
        --custom-dataset-type single_turn \
        --extra-inputs "max_tokens:${OSL}" \
        --extra-inputs "min_tokens:${OSL}" \
        --extra-inputs "ignore_eos:true" \
        --extra-inputs "stream:true" \
        --streaming \
        --ui none \
        --no-server-metrics
}

echo "======================================================================"
echo "  Dry Run: Qwen3-VL-30B Agg vs EPD on 4xB200"
echo "======================================================================"
echo "  Model:         $MODEL"
echo "  Input file:    $INPUT_FILE"
echo "  Requests:      $REQUEST_COUNT (warmup: $WARMUP_COUNT)"
echo "  Concurrency:   $CONCURRENCY"
echo "  OSL:           $OSL"
echo ""

RESULTS=()

for entry in "${CONFIGS[@]}"; do
    label="${entry%%:*}"
    script="${entry#*:}"

    echo "----------------------------------------------------------------------"
    echo "  [$label] Starting server..."
    echo "----------------------------------------------------------------------"

    # Launch server in its own process group (setsid)
    setsid bash "$script" --model-path "$MODEL" &
    SERVER_PID=$!

    if wait_for_model; then
        echo "  [$label] Running aiperf..."
        if run_aiperf; then
            RESULTS+=("PASS  $label")
            echo "  [$label] PASS"
        else
            RESULTS+=("FAIL  $label  (aiperf returned non-zero)")
            echo "  [$label] FAIL (aiperf error)"
        fi
    else
        RESULTS+=("FAIL  $label  (server startup timeout)")
        echo "  [$label] FAIL (timeout)"
    fi

    cleanup_server
done

echo ""
echo "======================================================================"
echo "  Dry Run Results"
echo "======================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""

# Exit non-zero if any config failed
for r in "${RESULTS[@]}"; do
    if [[ "$r" == FAIL* ]]; then
        exit 1
    fi
done
