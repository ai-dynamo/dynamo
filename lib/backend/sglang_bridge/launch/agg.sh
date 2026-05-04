#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the SGLang sidecar bridge. Three processes:
#   1. dynamo.frontend (HTTP :8000, KvRouter, etcd discovery)
#   2. python -m sglang.launch_server --grpc-mode (gRPC on :30000)
#   3. dynamo-sglang-bridge (Rust LLMEngine -> SGLang gRPC)
#
# This is the equivalent of examples/backends/sglang/launch/agg.sh, but
# replacing `python -m dynamo.sglang` with the Rust bridge talking to an
# upstream-bare SGLang via its gRPC schema.
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SGLANG_VENV="${SGLANG_VENV:-/ephemeral/sglang/.venv}"
BRIDGE_BIN="${BRIDGE_BIN:-/ephemeral/cargo-target/debug/dynamo-sglang-bridge}"
DYNAMO_VENV="${DYNAMO_VENV:-/ephemeral/dynamo-sglang-grpc/.venv}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-30000}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--model-path <name>]

Env:
  MODEL              HF repo or local path (default: $MODEL)
  SGLANG_VENV        venv that has stock upstream sglang + smg-grpc-servicer
                     (default: /ephemeral/sglang/.venv)
  DYNAMO_VENV        venv that has dynamo.frontend installed
                     (default: /ephemeral/dynamo-sglang-grpc/.venv)
  BRIDGE_BIN         path to dynamo-sglang-bridge binary
  SGLANG_GRPC_PORT   port sglang serves gRPC on (default 30000)
  DYN_HTTP_PORT      dynamo frontend HTTP port (default 8000)
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "ERROR: bridge binary not found at $BRIDGE_BIN" >&2
    echo "Build with: CARGO_TARGET_DIR=/ephemeral/cargo-target cargo build -p dynamo-sglang-bridge" >&2
    exit 1
fi

echo "=== Launching aggregated serving (SGLang sidecar bridge POC) ==="
echo "Model:           $MODEL"
echo "Frontend HTTP:   :$HTTP_PORT"
echo "SGLang gRPC:     :$SGLANG_GRPC_PORT"

# 1) Dynamo frontend
OTEL_SERVICE_NAME=dynamo-frontend \
"$DYNAMO_VENV/bin/python" -m dynamo.frontend &

# 2) Stock upstream SGLang in --grpc-mode (single port, no HTTP)
CUDA_VISIBLE_DEVICES=0 OTEL_SERVICE_NAME=sglang-server \
"$SGLANG_VENV/bin/python" -m sglang.launch_server \
    --grpc-mode \
    --model-path "$MODEL" \
    --port "$SGLANG_GRPC_PORT" \
    --tp 1 \
    --trust-remote-code \
    --disable-piecewise-cuda-graph &

# 3) Rust sidecar bridge — registers as Dynamo network worker via etcd, forwards
#    each PreprocessedRequest to SGLang's Generate RPC. Sleep gives SGLang a
#    head start so HealthCheck succeeds on first try.
sleep 5
OTEL_SERVICE_NAME=sglang-bridge DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8082}" \
RUST_LOG=info,dynamo_sglang_bridge=debug \
"$BRIDGE_BIN" \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --sglang-grpc-endpoint "http://127.0.0.1:$SGLANG_GRPC_PORT" &

wait -n
