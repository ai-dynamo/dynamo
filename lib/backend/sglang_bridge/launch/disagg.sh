#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode via the SGLang sidecar bridge. One prefill
# (sglang+bridge) + one decode (sglang+bridge), KV transferred via NIXL.
#
# Equivalent of examples/backends/sglang/launch/disagg.sh.
#
# The legacy gRPC schema (`sglang.grpc.scheduler`) carries `DisaggregatedParams`
# first-class on `GenerateRequest` (bootstrap_host/bootstrap_port/bootstrap_room).
# The bridge already forwards `PreprocessedRequest.bootstrap_info` into that
# proto field — no further changes needed.
#
# TODO: the bridge needs to advertise its disaggregation mode to Dynamo so the
# frontend's prefill/decode routers know which worker is which. Currently both
# bridges register identically; frontend cannot distinguish prefill from decode.
# Track as: extend bridge with `--disaggregation-mode {prefill,decode}` that
# is reflected in WorkerConfig.endpoint_types or similar.
#
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DISAGG_BOOTSTRAP_PORT="${DISAGG_BOOTSTRAP_PORT:-8998}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
BRIDGE_BIN="${BRIDGE_BIN:-/ephemeral/cargo-target/debug/dynamo-sglang-bridge}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    exit 1
fi

echo "=== Disaggregated PD (sidecar bridge POC) ==="
echo "Model:           $MODEL"
echo "Bootstrap port:  $DISAGG_BOOTSTRAP_PORT"

# Frontend
DYN_HTTP_PORT="$HTTP_PORT" OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend &

# Prefill worker on GPU 0
CUDA_VISIBLE_DEVICES=0 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30000 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --disable-piecewise-cuda-graph &

# Decode worker on GPU 1
CUDA_VISIBLE_DEVICES=1 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode --port 30001 \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --disable-piecewise-cuda-graph &

sleep 8

# Prefill bridge — registers with endpoint_types=prefill, generates bootstrap_room
# per request and yields it as the first chunk for the frontend to forward.
DYN_SYSTEM_PORT=8082 OTEL_SERVICE_NAME=sglang-bridge-prefill \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30000 \
    --disaggregation-mode prefill &

# Decode bridge — pulls bootstrap_info from incoming requests and forwards
# via DisaggregatedParams in the gRPC Generate call.
DYN_SYSTEM_PORT=8083 OTEL_SERVICE_NAME=sglang-bridge-decode \
RUST_LOG=info \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint http://127.0.0.1:30001 \
    --disaggregation-mode decode &

wait -n
