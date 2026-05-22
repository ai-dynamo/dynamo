#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the Rust bridge to SGLang's native gRPC server.
# Three processes: dynamo.frontend, sglang.launch_server, dynamo.sglang_grpc.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"

print_launch_banner "Launching Aggregated Serving (gRPC bridge)" "$MODEL" "$HTTP_PORT"

# dynamo.frontend's preprocessor needs a local dir to load the tokenizer from.
FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"

python3 -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

# --skip-tokenizer-init: dynamo.frontend is the sole tok/detok; wire carries token IDs.
SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --port "$SGLANG_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --trust-remote-code \
    --skip-tokenizer-init &

echo "Waiting for SGLang gRPC (:$SGLANG_GRPC_PORT)..."
wait_for_port "$SGLANG_GRPC_PORT" 600 \
    || { echo "ERROR: SGLang gRPC :$SGLANG_GRPC_PORT did not open within 600s" >&2; exit 1; }

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
python3 -m dynamo.sglang_grpc \
    --sglang-grpc-endpoint "http://127.0.0.1:$SGLANG_GRPC_PORT" &

wait_any_exit
