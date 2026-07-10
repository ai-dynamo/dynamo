#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the vLLM gRPC sidecar (1 GPU).
#
# Unlike agg.sh (which runs vLLM in-process inside the Dynamo worker), this
# launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. A native `vllm-rs serve` engine exposing the vLLM gRPC server
#   3. The Dynamo vLLM sidecar worker, which talks to (2) over vLLM gRPC
#
# `vllm-rs serve` is the Rust vLLM frontend: it self-manages a headless Python
# EngineCore subprocess and mounts the vLLM gRPC service on
# --grpc-port. The Dynamo sidecar (dynamo-vllm-sidecar) never imports
# vllm; it is given ONLY the vLLM gRPC endpoint and discovers the model, role,
# parallelism, and KV config from the vLLM gRPC API.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default model
MODEL="Qwen/Qwen3-0.6B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>       Specify model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional options are passed through to the vllm-rs serve engine."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
GRPC_HOST="${GRPC_HOST:-127.0.0.1}"
GRPC_PORT="${GRPC_PORT:-50051}"
# vllm-rs runs its own OpenAI HTTP frontend; it is unused by the sidecar but
# still binds, so keep it off the Dynamo frontend's port (8000).
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Native vLLM engine (Rust frontend + managed Python EngineCore) with the
# vLLM gRPC server. The model is positional; --enforce-eager and
# --max-num-seqs are forwarded to the Python EngineCore. --enforce-eager is for
# quick startup; drop it for production.
vllm-rs serve "$MODEL" \
    --port "$VLLM_RS_HTTP_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --grpc-port "$GRPC_PORT" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no vllm import; vLLM gRPC client only). It receives
# ONLY the endpoint; model and role are discovered from the engine.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-sidecar \
    --grpc-endpoint "${GRPC_HOST}:${GRPC_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
