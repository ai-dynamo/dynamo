#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the upstream SGLang SMG sidecar (1 GPU).
#
# This launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. Upstream SGLang with --grpc-mode exposing SMG's scheduler service
#   3. The Dynamo SGLang SMG sidecar worker, which talks to (2)
#
# The SGLang environment must include the optional gRPC dependencies:
#   pip install 'smg-grpc-servicer[sglang]>=0.5.2'
#
# v1 supports aggregated serving and SGLang disaggregated prefill/decode handoff.
# KV-aware routing remains disabled because upstream SMG does not expose KV event
# sources or rank load metrics.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_sglang_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="Qwen/Qwen3-0.6B"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model-path)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>       Specify model (default: $MODEL)"
            echo "  --model-path <name>  Alias for --model"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional options are passed through to python3 -m sglang.launch_server."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SMG_PORT="${SMG_PORT:-40000}"
SMG_CONNECTIONS="${SMG_CONNECTIONS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching SGLang SMG Sidecar Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "SMG endpoint: ${SGLANG_HOST}:${SMG_PORT}"

"$PYTHON_BIN" -m dynamo.frontend &

SGLANG_GRPC_PORT="$SMG_PORT" \
"$PYTHON_BIN" -m sglang.launch_server \
    --grpc-mode \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_HTTP_PORT" \
    --tp 1 \
    --trust-remote-code \
    --disable-cuda-graph \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-sglang-smg-sidecar \
    --smg-endpoint "${SGLANG_HOST}:${SMG_PORT}" \
    --smg-connections "$SMG_CONNECTIONS" &

wait_any_exit
