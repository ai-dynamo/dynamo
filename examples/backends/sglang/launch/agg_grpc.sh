#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via SGLang's embedded Dynamo backend.
# Two processes: dynamo.frontend, sglang.launch_server.
# GPUs: 1

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"

print_launch_banner "Launching Aggregated SGLang gRPC Serving" "$MODEL" "$HTTP_PORT"

require_sglang_enable_dynamo

# dynamo.frontend's preprocessor needs a local dir to load the tokenizer from.
FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"

trap 'echo Cleaning up...; kill 0' EXIT

python3 -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
python3 -m sglang.launch_server \
    --enable-dynamo \
    --port "$SGLANG_GRPC_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --page-size 16 \
    --trust-remote-code \
    --skip-tokenizer-init &

wait_any_exit
