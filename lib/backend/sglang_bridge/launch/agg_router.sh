#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two aggregated workers behind Dynamo's KV-aware router, talking to SGLang
# via the sidecar bridge.
#
# Equivalent of examples/backends/sglang/launch/agg_router.sh but with the
# Rust bridge replacing python -m dynamo.sglang.
#
# NOTE: KV-aware routing currently falls back to round-robin because the
# bridge does not yet subscribe to SGLang's `SubscribeKvEvents` gRPC stream.
# Wiring that is POC-2 work; in the meantime this script exercises the
# multi-worker dispatch path.
#
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
BRIDGE_BIN="${BRIDGE_BIN:-/ephemeral/cargo-target/debug/dynamo-sglang-bridge}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    exit 1
fi

echo "=== KV-router (round-robin until SubscribeKvEvents wired) ==="
echo "Model:           $MODEL"
echo "Frontend HTTP:   :$HTTP_PORT, --router-mode kv"

# Frontend with KV router enabled
DYN_HTTP_PORT="$HTTP_PORT" OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend --router-mode kv &

# Worker 0 — SGLang :30000 + bridge sys-port 8082
CUDA_VISIBLE_DEVICES=0 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30000 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disable-piecewise-cuda-graph &

# Worker 1 — SGLang :30001 + bridge sys-port 8083
CUDA_VISIBLE_DEVICES=1 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30001 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disable-piecewise-cuda-graph &

# Bridge sidecars — 1 per SGLang instance. Both register against the same
# Dynamo `dynamo.backend.generate` endpoint name; KvRouter sees two distinct
# worker_ids and routes between them.
sleep 8

DYN_SYSTEM_PORT=8082 OTEL_SERVICE_NAME=sglang-bridge-0 \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30000 &

DYN_SYSTEM_PORT=8083 OTEL_SERVICE_NAME=sglang-bridge-1 \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30001 &

wait -n
