#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode via the Rust sidecar bridge to SGLang's
# native gRPC server (`sglang.runtime.v1`).
#
# Layout:
#   dynamo.frontend (HTTP :8000)
#     ├─> prefill bridge (component=prefill)
#     │     └─> SGLang --enable-grpc :40000 (GPU 0, --disaggregation-mode prefill)
#     └─> decode bridge  (component=backend)
#           └─> SGLang --enable-grpc :40001 (GPU 1, --disaggregation-mode decode)
#
# KV transfer runs over the configured backend (NIXL by default).
# `DisaggregatedParams.bootstrap_room` is int64, preserving the
# PrefillRouter's dp_rank encoding across the wire.
#
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WORKSPACE_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-8998}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-30000}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-30001}"
PREFILL_GRPC_PORT="${PREFILL_GRPC_PORT:-40000}"
DECODE_GRPC_PORT="${DECODE_GRPC_PORT:-40001}"
TRANSFER_BACKEND="${SGLANG_TRANSFER_BACKEND:-nixl}"
BRIDGE_BIN="${BRIDGE_BIN:-$(command -v dynamo-sglang-bridge || echo "$WORKSPACE_ROOT/target/debug/dynamo-sglang-bridge")}"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: dynamo-sglang-bridge not found at $BRIDGE_BIN" >&2
    echo "Build it: (cd $WORKSPACE_ROOT && cargo build -p dynamo-sglang-bridge)" >&2
    exit 1
fi

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

print_launch_banner "Launching Disaggregated Serving (gRPC bridge, 2 GPUs)" "$MODEL" "$HTTP_PORT"

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

CUDA_VISIBLE_DEVICES=0 SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$PREFILL_GRPC_PORT" \
    --port "$PREFILL_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --trust-remote-code \
    --tokenizer-worker-num 1 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --disable-piecewise-cuda-graph &

CUDA_VISIBLE_DEVICES=1 SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$DECODE_GRPC_PORT" \
    --port "$DECODE_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --trust-remote-code \
    --tokenizer-worker-num 1 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --disable-piecewise-cuda-graph &

echo "Waiting for SGLang gRPC (:$PREFILL_GRPC_PORT, :$DECODE_GRPC_PORT)..."
for port in "$PREFILL_GRPC_PORT" "$DECODE_GRPC_PORT"; do
    if ! wait_for_port "$port" 180; then
        echo "ERROR: SGLang gRPC port :$port did not open within 180s" >&2
        exit 1
    fi
    echo "  :$port open"
done

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} OTEL_SERVICE_NAME=sglang-bridge-prefill \
"$BRIDGE_BIN" \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$PREFILL_GRPC_PORT" \
    --disaggregation-mode prefill &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} OTEL_SERVICE_NAME=sglang-bridge-decode \
"$BRIDGE_BIN" \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$DECODE_GRPC_PORT" \
    --disaggregation-mode decode &

wait_any_exit
