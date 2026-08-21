#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving through TensorRT-LLM's native gRPC server (1 GPU).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model-path)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for $1"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model|--model-path <name>] [TensorRT-LLM engine options...]"
            echo
            echo "Additional options are passed to the TensorRT-LLM engine."
            echo
            echo "Environment overrides:"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  TRTLLM_PYTHON           Python with TensorRT-LLM installed (default: python3)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment (default: 0)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  TRTLLM_GRPC_PORT        TensorRT-LLM gRPC port (default: 50051)"
            echo "  TRTLLM_CONTEXT_LENGTH   Registered model context length (default: 4096)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

TRTLLM_EXTRA_CONFIG=""
trtllm_exit_trap() {
    local rc=$?
    if [[ -n "$TRTLLM_EXTRA_CONFIG" ]]; then
        rm -f -- "$TRTLLM_EXTRA_CONFIG"
    fi
    echo "Cleaning up..."
    dynamo_reap_and_exit "$rc"
}
trap trtllm_exit_trap EXIT

TRTLLM_PYTHON="${TRTLLM_PYTHON:-python3}"
TRTLLM_GRPC_PORT="${TRTLLM_GRPC_PORT:-50051}"
TRTLLM_CONTEXT_LENGTH="${TRTLLM_CONTEXT_LENGTH:-4096}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_trtllm_override_args_with_mem)
TRTLLM_GPU_MEM_ARGS=()
if [[ -n "$GPU_MEM_ARGS" ]]; then
    TRTLLM_EXTRA_CONFIG=$(mktemp "${TMPDIR:-/tmp}/dynamo-trtllm-sidecar.XXXXXX.yaml")
    printf '%s\n' "$GPU_MEM_ARGS" > "$TRTLLM_EXTRA_CONFIG"
    TRTLLM_GPU_MEM_ARGS=(--extra_llm_api_options "$TRTLLM_EXTRA_CONFIG")
fi

print_launch_banner "Launching TensorRT-LLM Native-gRPC Sidecar (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "TensorRT-LLM gRPC: 127.0.0.1:${TRTLLM_GRPC_PORT}" \
    "Context length:    ${TRTLLM_CONTEXT_LENGTH}"

python3 -m dynamo.frontend &

# TensorRT-LLM's native gRPC listener is unauthenticated; keep it on loopback.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc \
    --host 127.0.0.1 \
    --port "$TRTLLM_GRPC_PORT" \
    "${TRTLLM_GPU_MEM_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    dynamo-trtllm-sidecar \
    --trtllm-endpoint "127.0.0.1:${TRTLLM_GRPC_PORT}" \
    --model-path "$MODEL" \
    --context-length "$TRTLLM_CONTEXT_LENGTH" &

wait_any_exit
