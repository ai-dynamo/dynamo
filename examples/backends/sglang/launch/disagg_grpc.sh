#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode via SGLang's embedded Dynamo backend.
# Three processes: dynamo.frontend, sglang prefill, sglang decode.
# GPUs: 2

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DISAGG_BOOTSTRAP_PORT="${DISAGG_BOOTSTRAP_PORT:-8998}"
PREFILL_GRPC_PORT="${PREFILL_GRPC_PORT:-40000}"
DECODE_GRPC_PORT="${DECODE_GRPC_PORT:-40002}"
PREFILL_SYSTEM_PORT="${PREFILL_SYSTEM_PORT:-8081}"
DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-8082}"
PREFILL_COMPONENT="${PREFILL_COMPONENT:-prefill}"
DECODE_COMPONENT="${DECODE_COMPONENT:-backend}"

print_launch_banner "Launching Disaggregated SGLang gRPC Serving (2 GPUs)" "$MODEL" "$HTTP_PORT"

require_sglang_enable_dynamo

FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"

trap 'echo Cleaning up...; kill 0' EXIT

python3 -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

for role in prefill decode; do
    if [ "$role" = "prefill" ]; then
        gpu=0; grpc_port=$PREFILL_GRPC_PORT
        system_port=$PREFILL_SYSTEM_PORT
        component=$PREFILL_COMPONENT
    else
        gpu=1; grpc_port=$DECODE_GRPC_PORT
        system_port=$DECODE_SYSTEM_PORT
        component=$DECODE_COMPONENT
    fi
    CUDA_VISIBLE_DEVICES=$gpu \
    DYN_SYSTEM_PORT="$system_port" \
    DYN_COMPONENT="$component" \
    python3 -m sglang.launch_server \
        --enable-dynamo \
        --port "$grpc_port" \
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

wait_any_exit
