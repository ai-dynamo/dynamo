#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving through vLLM's native gRPC server (1 GPU).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"   # build_vllm_gpu_mem_args
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [vLLM engine options...]"
            echo
            echo "Additional options are passed to the managed vLLM engine."
            echo
            echo "Environment overrides:"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment (default: 0)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  VLLM_RS_HTTP_PORT       vLLM HTTP port (default: 8100)"
            echo "  VLLM_GRPC_PORT          vLLM gRPC port (default: 50051)"
            echo "  MAX_MODEL_LEN           Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS     Maximum concurrent sequences (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES  KV cache cap when not profiling (default: 1119388000)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"
VLLM_GRPC_PORT="${VLLM_GRPC_PORT:-50051}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Default KV cache cap from profiling (2x safety over min=560 MiB); ~3.8 GiB peak VRAM.
# The profiler/test framework takes precedence through _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching vLLM Native-gRPC Sidecar (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "vLLM HTTP:  http://127.0.0.1:${VLLM_RS_HTTP_PORT}" \
    "vLLM gRPC:  127.0.0.1:${VLLM_GRPC_PORT}"

python -m dynamo.frontend &

# vllm-rs manages the headless Python engine and exposes native gRPC on loopback.
# Arguments after -- are forwarded to the managed engine.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_RS_HTTP_PORT" \
    --grpc-port "$VLLM_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-sidecar \
    --vllm-endpoint "127.0.0.1:${VLLM_GRPC_PORT}" &

wait_any_exit
