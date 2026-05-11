#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving against SGLang's native gRPC server (alexnails Phase 1
# stack + our proto additions). No smg-grpc-servicer in the path.
#
# Layout:
#   dynamo.frontend (HTTP :8000)
#     └─> bridge (Dynamo worker, in-process via dynamo-sglang-bridge crate)
#           └─> SGLang --enable-grpc (native tonic server on :40000)
#
# Requires:
#   - SGLang fork on idhanani/alexnails-on-main with sglang_grpc wheel installed
#     (cd /ephemeral/sglang/rust/sglang-grpc && maturin develop)
#   - Dynamo bridge built against sglang.runtime.v1 proto
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
BRIDGE_BIN="${BRIDGE_BIN:-/ephemeral/cargo-target/debug/dynamo-sglang-bridge}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    exit 1
fi

echo "=== Native gRPC aggregated path ==="
echo "Model:          $MODEL"
echo "Frontend HTTP:  :$HTTP_PORT"
echo "SGLang HTTP:    :$SGLANG_HTTP_PORT"
echo "SGLang gRPC:    :$SGLANG_GRPC_PORT  (sglang.runtime.v1)"

# Dynamo frontend
DYN_HTTP_PORT="$HTTP_PORT" OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend &

# SGLang with native gRPC enabled (alongside HTTP). Single-process tokenizer
# worker — alexnails' server doesn't yet support tokenizer-worker-num > 1.
SGLANG_ENABLE_GRPC=1 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --port "$SGLANG_HTTP_PORT" \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --tokenizer-worker-num 1 &

# Wait for gRPC port to bind. The native server starts after the scheduler
# is up, so this takes the same ~30-60s as the smg path.
echo "Waiting for SGLang native gRPC (:$SGLANG_GRPC_PORT)..."
for _ in $(seq 1 120); do
    if (echo > /dev/tcp/127.0.0.1/$SGLANG_GRPC_PORT) 2>/dev/null; then
        echo "  :$SGLANG_GRPC_PORT open"
        break
    fi
    sleep 1
done

# Bridge — registers as a normal Dynamo worker, talks to the native gRPC
# server. No --disaggregation-mode here (agg path).
DYN_SYSTEM_PORT=8082 OTEL_SERVICE_NAME=sglang-bridge \
DYN_LOG=info,dynamo_sglang_bridge=debug \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$SGLANG_GRPC_PORT" &

wait -n
