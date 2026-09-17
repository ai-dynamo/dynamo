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
            echo "  TRTLLM_CONTEXT_LENGTH   Model context length, applied to both the engine and the"
            echo "                          sidecar (default: 4096; unset when --max_seq_len is given)"
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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Keep the engine and the sidecar on one number. Started without `--max_seq_len`,
# TensorRT-LLM reports its `max_input_len` default instead of a context length
# and the sidecar discards it, so pass the same value to both. When the caller
# supplies `--max_seq_len`, theirs wins and the sidecar adopts the engine's
# report rather than overriding it with a default it was never told about.
TRTLLM_MAX_SEQ_LEN_ARGS=()
TRTLLM_CONTEXT_LENGTH_ARGS=()
trtllm_max_seq_len_supplied=0
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --max_seq_len|--max_seq_len=*) trtllm_max_seq_len_supplied=1 ;;
    esac
done
if [[ "$trtllm_max_seq_len_supplied" -eq 0 ]]; then
    TRTLLM_CONTEXT_LENGTH="${TRTLLM_CONTEXT_LENGTH:-4096}"
    TRTLLM_MAX_SEQ_LEN_ARGS=(--max_seq_len "$TRTLLM_CONTEXT_LENGTH")
fi
if [[ -n "$TRTLLM_CONTEXT_LENGTH" ]]; then
    TRTLLM_CONTEXT_LENGTH_ARGS=(--context-length "$TRTLLM_CONTEXT_LENGTH")
fi

# `--grpc` needs `smg-grpc-proto`, which TRT-LLM keeps behind its optional
# `grpc-smg` extra. Constraint copied from that extra so we resolve what
# upstream resolves.
if ! "$TRTLLM_PYTHON" -c "import smg_grpc_proto" >/dev/null 2>&1; then
    "$TRTLLM_PYTHON" -m pip install --no-cache-dir "smg-grpc-proto>=0.4.2"
fi

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
    "Context length:    ${TRTLLM_CONTEXT_LENGTH:-from engine report}"

python3 -m dynamo.frontend &

# TensorRT-LLM's native gRPC listener is unauthenticated; keep it on loopback.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$TRTLLM_PYTHON" -m tensorrt_llm.commands.serve "$MODEL" \
    --grpc \
    --host 127.0.0.1 \
    --port "$TRTLLM_GRPC_PORT" \
    "${TRTLLM_MAX_SEQ_LEN_ARGS[@]}" \
    "${TRTLLM_GPU_MEM_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    dynamo-trtllm-sidecar \
    --grpc-endpoint "127.0.0.1:${TRTLLM_GRPC_PORT}" \
    --model-path "$MODEL" \
    "${TRTLLM_CONTEXT_LENGTH_ARGS[@]}" &

wait_any_exit
