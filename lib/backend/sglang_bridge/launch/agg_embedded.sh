#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# B1 architecture: Dynamo frontend with the sglang_bridge crate linked
# in-process as an InProcessTokens engine. No bridge sidecar, no Dynamo
# IPC hop.  Two processes total:
#   1. dynamo.frontend with DYN_ENGINE_TYPE=sglang-grpc  (HTTP :8000,
#      tokenizes locally, dials SGLang gRPC directly)
#   2. python -m sglang.launch_server --grpc-mode  (:30000)
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
# Frontend's --model-path validation requires a local dir, so for HF ids
# we point at the cached snapshot to pick up the tokenizer. The bridge
# itself does not need weights.
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-30000}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

echo "=== Launching B1 (frontend-embedded sglang bridge) ==="
echo "Model:          $MODEL"
echo "Frontend HTTP:  :$HTTP_PORT"
echo "SGLang gRPC:    :$SGLANG_GRPC_PORT"

# 1) Stock upstream SGLang in --grpc-mode
CUDA_VISIBLE_DEVICES=0 OTEL_SERVICE_NAME=sglang-server \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode \
    --model-path "$MODEL" \
    --port "$SGLANG_GRPC_PORT" \
    --tp 1 \
    --trust-remote-code \
    --disable-piecewise-cuda-graph &

# 2) Dynamo frontend with the bridge embedded in-process. SGLANG_GRPC_ENDPOINT
#    drives the bridge's gRPC client; DYN_ENGINE_TYPE swaps the launcher's
#    EngineConfig::Dynamic for EngineConfig::InProcessTokens.
#
# Wait for SGLang's gRPC port to accept connections before bringing up the
# frontend — the bridge calls HealthCheck on start(), and a stock SGLang
# load takes 30-60s.
echo "Waiting for SGLang gRPC port :$SGLANG_GRPC_PORT ..."
for _ in $(seq 1 120); do
    if (echo > /dev/tcp/127.0.0.1/$SGLANG_GRPC_PORT) 2>/dev/null; then
        echo "SGLang port open"
        break
    fi
    sleep 1
done

OTEL_SERVICE_NAME=dynamo-frontend \
DYN_HTTP_PORT="$HTTP_PORT" \
DYN_ENGINE_TYPE=sglang-grpc \
SGLANG_GRPC_ENDPOINT="http://127.0.0.1:$SGLANG_GRPC_PORT" \
RUST_LOG=info,dynamo_sglang_bridge=debug \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$MODEL_LOCAL_PATH" &

wait -n
