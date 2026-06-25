#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the upstream vLLM SMG sidecar (1 GPU).
#
# This launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. Upstream `vllm serve --grpc` exposing SMG's vLLM gRPC service
#   3. The Dynamo vLLM SMG sidecar worker, which talks to (2)
#
# The vLLM environment must include the optional gRPC dependencies:
#   pip install 'vllm[grpc]'
# or equivalently:
#   pip install 'smg-grpc-servicer[vllm]>=0.5.2'
#
# v1 scope is aggregated text/token generation. Disaggregated serving,
# multimodal URL passthrough, KV-aware routing, and logprobs are intentionally
# not enabled on this SMG path.

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
            echo "Additional options are passed through to vllm serve --grpc."
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
SMG_HOST="${SMG_HOST:-127.0.0.1}"
SMG_PORT="${SMG_PORT:-50051}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching SMG Sidecar Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
"$PYTHON_BIN" -m dynamo.frontend &

# 2. Upstream vLLM engine with SMG gRPC enabled. The vLLM OpenAI HTTP server is
# not started in this mode; --port is the gRPC listener port when --grpc is set.
vllm serve "$MODEL" \
    --grpc \
    --host "$SMG_HOST" \
    --port "$SMG_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo SMG sidecar worker (no vllm import; SMG client only).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-smg-sidecar \
    --smg-endpoint "${SMG_HOST}:${SMG_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
