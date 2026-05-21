#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-machine agg via the embedded gRPC bridge (no sidecar binary).
# Frontend dials SGLang `--enable-grpc` directly. Multi-worker setups use
# the sidecar binary instead — see agg_grpc.sh.
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-1}"
# dynamo.frontend's sglang-grpc engine requires a local model dir, not an HF id.
FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"

print_launch_banner "Launching Aggregated Serving (embedded gRPC bridge)" "$MODEL" "$HTTP_PORT"

# Start SGLang before the frontend so its 120s HealthCheck retry budget
# isn't burned waiting for the listener.
SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --port "$SGLANG_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp "$TP" \
    --trust-remote-code \
    --enable-metrics \
    --tokenizer-worker-num 1 &

echo "Waiting for SGLang gRPC (:$SGLANG_GRPC_PORT)..."
if ! wait_for_port "$SGLANG_GRPC_PORT" 180; then
    echo "ERROR: SGLang gRPC port :$SGLANG_GRPC_PORT did not open within 180s" >&2
    exit 1
fi
echo "  :$SGLANG_GRPC_PORT open"

SGLANG_GRPC_ENDPOINT="http://127.0.0.1:$SGLANG_GRPC_PORT" \
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --dyn-engine sglang-grpc \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

wait_any_exit
