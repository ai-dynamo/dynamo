#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUEST_COUNT="${REQUEST_COUNT:-100}"
OSL="${OSL:-70}"
INPUT_FILE="${INPUT_FILE:-$SCRIPT_DIR/.data/qwen3_vl_9image_100req.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/logs/qwen3_vl_custom_encoder/conc${CONCURRENCY}}"

if [[ ! -f "$INPUT_FILE" ]]; then
    python "$SCRIPT_DIR/generate_workload.py" \
        --model "$MODEL_NAME" \
        --request-count "$REQUEST_COUNT"
fi

mkdir -p "$ARTIFACT_DIR"

aiperf profile \
    --model "$MODEL_NAME" \
    --url "$SERVER_URL" \
    --endpoint-type chat \
    --request-count "$REQUEST_COUNT" \
    --warmup-request-count 2 \
    --concurrency "$CONCURRENCY" \
    --osl "$OSL" \
    --extra-inputs ignore_eos:true \
    --input-file "$INPUT_FILE" \
    --custom-dataset-type single_turn \
    --artifact-dir "$ARTIFACT_DIR" \
    --use-server-token-count \
    --no-server-metrics \
    --ui simple
