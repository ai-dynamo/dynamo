#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving through SGLang's native gRPC server (1 GPU).
# Requires an SGLang build with native gRPC sidecar support.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"   # build_sglang_gpu_mem_args
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
            echo "Usage: $0 [--model|--model-path <name>] [SGLang engine options...]"
            echo
            echo "Additional options are passed to the SGLang engine."
            echo
            echo "Environment overrides:"
            echo "  MODEL                   Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  SGLANG_PYTHON           Python with SGLang installed (default: python3)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment (default: 0)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  SGLANG_HTTP_PORT        SGLang HTTP port (default: 30000)"
            echo "  SGLANG_GRPC_PORT        SGLang gRPC port (default: 30001)"
            echo "  MAX_MODEL_LEN           Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS     Maximum concurrent sequences (default: 2)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
SGLANG_HOST="127.0.0.1"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-30001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner "Launching SGLang Native-gRPC Sidecar (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "SGLang HTTP: http://${SGLANG_HOST}:${SGLANG_HTTP_PORT}" \
    "SGLang gRPC: ${SGLANG_HOST}:${SGLANG_GRPC_PORT}"

python3 -m dynamo.frontend &

# --grpc-port enables the native Rust gRPC server alongside SGLang's HTTP API.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_HTTP_PORT" \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --incremental-streaming-output \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    dynamo-sglang-sidecar \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_GRPC_PORT}" &

wait_any_exit
