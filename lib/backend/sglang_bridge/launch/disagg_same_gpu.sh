#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated PD with both workers on a single GPU. Useful for debugging
# disagg flows on a single-GPU box.
#
# GPUs: 1 (workers share the same physical GPU; reduce mem fraction)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DISAGG_BOOTSTRAP_PORT="${DISAGG_BOOTSTRAP_PORT:-8998}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
BRIDGE_BIN="${BRIDGE_BIN:-/ephemeral/cargo-target/debug/dynamo-sglang-bridge}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MEM_FRACTION="${MEM_FRACTION:-0.42}"   # ~half of single-GPU memory per worker

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    exit 1
fi

echo "=== Disaggregated PD on a single GPU (sidecar bridge POC) ==="

DYN_HTTP_PORT="$HTTP_PORT" OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend &

CUDA_VISIBLE_DEVICES=0 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30000 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-piecewise-cuda-graph &

CUDA_VISIBLE_DEVICES=0 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30001 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-piecewise-cuda-graph &

sleep 10

DYN_SYSTEM_PORT=8082 OTEL_SERVICE_NAME=sglang-bridge-prefill \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30000 &

DYN_SYSTEM_PORT=8083 OTEL_SERVICE_NAME=sglang-bridge-decode \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30001 &

wait -n
