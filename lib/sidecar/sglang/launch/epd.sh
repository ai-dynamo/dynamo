#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Native SGLang encoder + language server(s), with Dynamo routing P/PD and D.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091
source "$DYNAMO_HOME/examples/common/launch_utils.sh"

DISAGGREGATED=false
if [[ "${1:-}" == "--disaggregated" ]]; then
    DISAGGREGATED=true
    shift
fi
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [--disaggregated] [SGLang language-server options...]"
    echo "Default: E+PD (2 GPUs). --disaggregated: E+P+D (3 GPUs)."
    echo "Set MODEL, SGLANG_PYTHON, SGLANG_ENCODER_GPU, SGLANG_PREFILL_GPU,"
    echo "SGLANG_DECODE_GPU, ENCODER_TRANSFER_BACKEND, and DYN_HTTP_PORT as needed."
    exit 0
fi
trap dynamo_exit_trap EXIT

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
ENCODER_PORT="${SGLANG_ENCODER_HTTP_PORT:-30000}"
PREFILL_PORT="${SGLANG_PREFILL_HTTP_PORT:-30010}"
PREFILL_GRPC_PORT="${SGLANG_PREFILL_GRPC_PORT:-30011}"
DECODE_PORT="${SGLANG_DECODE_HTTP_PORT:-30020}"
DECODE_GRPC_PORT="${SGLANG_DECODE_GRPC_PORT:-30021}"
BOOTSTRAP_PORT="${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"
ENCODER_TRANSFER_BACKEND="${ENCODER_TRANSFER_BACKEND:-zmq_to_scheduler}"

# No encoder sidecar or --route-to-encoder: native SGLang owns the E hop.
CUDA_VISIBLE_DEVICES="${SGLANG_ENCODER_GPU:-0}" \
    "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$ENCODER_PORT" \
    --encoder-only --encoder-transfer-backend "$ENCODER_TRANSFER_BACKEND" &

PREFILL_ARGS=()
if $DISAGGREGATED; then
    PREFILL_ARGS=(--disaggregation-mode prefill
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
        --disaggregation-transfer-backend nixl)
fi
CUDA_VISIBLE_DEVICES="${SGLANG_PREFILL_GPU:-1}" \
    "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 \
    --port "$PREFILL_PORT" --grpc-port "$PREFILL_GRPC_PORT" \
    --incremental-streaming-output --language-only \
    --encoder-urls "http://127.0.0.1:$ENCODER_PORT" \
    --encoder-transfer-backend "$ENCODER_TRANSFER_BACKEND" \
    --context-length 8192 --max-total-tokens 8192 --max-running-requests 4 \
    "${PREFILL_ARGS[@]}" "$@" &
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" dynamo-sglang-sidecar \
    --grpc-endpoint "127.0.0.1:$PREFILL_GRPC_PORT" --bootstrap-host 127.0.0.1 &

if $DISAGGREGATED; then
    # Match native SGLang: only P dispatches to E; D receives KV from P.
    CUDA_VISIBLE_DEVICES="${SGLANG_DECODE_GPU:-2}" \
        "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL" --host 127.0.0.1 \
        --port "$DECODE_PORT" --grpc-port "$DECODE_GRPC_PORT" \
        --incremental-streaming-output --disaggregation-mode decode \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --disaggregation-transfer-backend nixl \
        --context-length 8192 --max-total-tokens 8192 --max-running-requests 4 \
        "$@" &
    DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" dynamo-sglang-sidecar \
        --grpc-endpoint "127.0.0.1:$DECODE_GRPC_PORT" &
fi

python3 -m dynamo.frontend &
wait_any_exit
