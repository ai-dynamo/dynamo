#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode against SGLang's native gRPC server
# (sglang.runtime.v1 — alexnails Phase 1 stack + our additions). NO smg
# in the inference path.
#
# Layout:
#   dynamo.frontend (HTTP :8000)
#     ├─> prefill bridge (component=prefill)
#     │     └─> SGLang --enable-grpc :40000 (GPU 0, --disaggregation-mode prefill)
#     │
#     └─> decode bridge  (component=backend)
#           └─> SGLang --enable-grpc :40001 (GPU 1, --disaggregation-mode decode)
#
# KV transfer via NIXL between the two SGLang instances. The proto's
# DisaggregatedParams.bootstrap_room is int64 (our addition) so Dynamo's
# compute_bootstrap_room dp_rank encoding survives the wire.
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

# SGLang HTTP ports are 30000/30001 (legacy positional); native gRPC defaults
# to HTTP + 10000, so 40000/40001.
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-30000}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-30001}"
PREFILL_GRPC_PORT="${PREFILL_GRPC_PORT:-40000}"
DECODE_GRPC_PORT="${DECODE_GRPC_PORT:-40001}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    exit 1
fi

echo "=== Disaggregated PD on sglang.runtime.v1 (no smg) ==="
echo "Model:              $MODEL"
echo "Frontend HTTP:      :$HTTP_PORT"
echo "Prefill SGLang gRPC: :$PREFILL_GRPC_PORT  (HTTP :$PREFILL_HTTP_PORT)"
echo "Decode  SGLang gRPC: :$DECODE_GRPC_PORT   (HTTP :$DECODE_HTTP_PORT)"
echo "Bootstrap rendezvous port: $DISAGG_BOOTSTRAP_PORT"

# Frontend
DYN_HTTP_PORT="$HTTP_PORT" OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend &

# Prefill SGLang on GPU 0
CUDA_VISIBLE_DEVICES=0 SGLANG_ENABLE_GRPC=1 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$PREFILL_GRPC_PORT" \
    --port "$PREFILL_HTTP_PORT" \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --tokenizer-worker-num 1 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --disable-piecewise-cuda-graph &

# Decode SGLang on GPU 1
CUDA_VISIBLE_DEVICES=1 SGLANG_ENABLE_GRPC=1 \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$DECODE_GRPC_PORT" \
    --port "$DECODE_HTTP_PORT" \
    --model-path "$MODEL" --tp 1 --trust-remote-code \
    --tokenizer-worker-num 1 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    --disable-piecewise-cuda-graph &

# Wait for both gRPC ports to bind. SGLang's native server starts the
# listener early; the bridge's HealthCheck retry covers the scheduler
# warmup window (~30-60s for Qwen3-0.6B).
echo "Waiting for SGLang native gRPC (:$PREFILL_GRPC_PORT, :$DECODE_GRPC_PORT)..."
for port in $PREFILL_GRPC_PORT $DECODE_GRPC_PORT; do
    for _ in $(seq 1 180); do
        if (echo > /dev/tcp/127.0.0.1/$port) 2>/dev/null; then
            echo "  :$port open"
            break
        fi
        sleep 1
    done
done

# Prefill bridge — registers as component=prefill (auto-derived from
# --disaggregation-mode prefill).
DYN_SYSTEM_PORT=8082 OTEL_SERVICE_NAME=sglang-bridge-prefill \
DYN_LOG=info,dynamo_sglang_bridge=debug \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$PREFILL_GRPC_PORT" \
    --disaggregation-mode prefill &

# Decode bridge.
DYN_SYSTEM_PORT=8083 OTEL_SERVICE_NAME=sglang-bridge-decode \
DYN_LOG=info,dynamo_sglang_bridge=debug \
"$BRIDGE_BIN" \
    --model-path "$MODEL" --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$DECODE_GRPC_PORT" \
    --disaggregation-mode decode &

wait -n
