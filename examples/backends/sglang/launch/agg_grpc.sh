#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the Rust sidecar bridge to SGLang's native gRPC
# server (`sglang.runtime.v1`, enabled by `sglang.launch_server --enable-grpc`).
#
# Layout:
#   dynamo.frontend (HTTP :8000)
#     └─> dynamo-sglang-bridge (Dynamo worker)
#           └─> SGLang --enable-grpc (gRPC :40000)
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WORKSPACE_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"
BRIDGE_BIN="${BRIDGE_BIN:-$(command -v dynamo-sglang-bridge || echo "$WORKSPACE_ROOT/target/debug/dynamo-sglang-bridge")}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: dynamo-sglang-bridge not found at $BRIDGE_BIN" >&2
    echo "Build it: (cd $WORKSPACE_ROOT && cargo build -p dynamo-sglang-bridge)" >&2
    exit 1
fi

# Wait up to TIMEOUT seconds for a TCP port on 127.0.0.1 to accept connections.
wait_for_port() {
    local port=$1 timeout=${2:-180}
    for _ in $(seq 1 "$timeout"); do
        if (echo > /dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

print_launch_banner "Launching Aggregated Serving (gRPC bridge)" "$MODEL" "$HTTP_PORT"

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# Single tokenizer worker: SGLang's native gRPC server does not yet support
# --tokenizer-worker-num > 1.
SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --port "$SGLANG_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --trust-remote-code \
    --tokenizer-worker-num 1 &

echo "Waiting for SGLang gRPC (:$SGLANG_GRPC_PORT)..."
if ! wait_for_port "$SGLANG_GRPC_PORT" 180; then
    echo "ERROR: SGLang gRPC port :$SGLANG_GRPC_PORT did not open within 180s" >&2
    exit 1
fi
echo "  :$SGLANG_GRPC_PORT open"

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} OTEL_SERVICE_NAME=sglang-bridge \
"$BRIDGE_BIN" \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$SGLANG_GRPC_PORT" &

wait_any_exit
