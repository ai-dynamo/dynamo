#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode via the Rust bridge to SGLang's gRPC server.
# Five processes: dynamo.frontend, sglang prefill, sglang decode, bridge prefill, bridge decode.
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DISAGG_BOOTSTRAP_PORT="${DISAGG_BOOTSTRAP_PORT:-8998}"
PREFILL_GRPC_PORT="${PREFILL_GRPC_PORT:-40000}"
DECODE_GRPC_PORT="${DECODE_GRPC_PORT:-40001}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-30000}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-30001}"
PREFILL_SYSTEM_PORT="${PREFILL_SYSTEM_PORT:-8081}"
DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-8082}"

print_launch_banner "Launching Disaggregated Serving (gRPC bridge, 2 GPUs)" "$MODEL" "$HTTP_PORT"

FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"

python3 -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

for role in prefill decode; do
    if [ "$role" = "prefill" ]; then
        gpu=0; grpc_port=$PREFILL_GRPC_PORT; http_port=$PREFILL_HTTP_PORT
    else
        gpu=1; grpc_port=$DECODE_GRPC_PORT; http_port=$DECODE_HTTP_PORT
    fi
    CUDA_VISIBLE_DEVICES=$gpu SGLANG_ENABLE_GRPC=1 \
    python3 -m sglang.launch_server \
        --enable-grpc \
        --grpc-port "$grpc_port" \
        --port "$http_port" \
        --model-path "$MODEL" \
        --tp 1 \
        --page-size 16 \
        --trust-remote-code \
        --skip-tokenizer-init \
        --disaggregation-mode "$role" \
        --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
        --disaggregation-transfer-backend nixl \
        --disable-piecewise-cuda-graph &
done

echo "Waiting for SGLang gRPC (:$PREFILL_GRPC_PORT, :$DECODE_GRPC_PORT)..."
for port in "$PREFILL_GRPC_PORT" "$DECODE_GRPC_PORT"; do
    wait_for_port "$port" 600 \
        || { echo "ERROR: SGLang gRPC :$port did not open within 600s" >&2; exit 1; }
done

DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT" \
python3 -m dynamo.sglang_grpc \
    --sglang-grpc-endpoint "http://127.0.0.1:$PREFILL_GRPC_PORT" &

DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" \
python3 -m dynamo.sglang_grpc \
    --sglang-grpc-endpoint "http://127.0.0.1:$DECODE_GRPC_PORT" &

wait_any_exit
