#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving through two SGLang native gRPC servers (2 GPUs).
# Requires SGLang PR #25185 plus server-side DisaggregatedParams forwarding.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model-path"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-path <name>] [SGLang options...]"
            echo
            echo "Set SGLANG_PYTHON to the Python executable containing the patched SGLang build."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_BOOTSTRAP_HOST="${SGLANG_BOOTSTRAP_HOST:-$SGLANG_HOST}"
SGLANG_DISAGGREGATION_BOOTSTRAP_PORT="${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"
SGLANG_PREFILL_HTTP_PORT="${SGLANG_PREFILL_HTTP_PORT:-30000}"
SGLANG_PREFILL_GRPC_PORT="${SGLANG_PREFILL_GRPC_PORT:-30001}"
SGLANG_DECODE_HTTP_PORT="${SGLANG_DECODE_HTTP_PORT:-30010}"
SGLANG_DECODE_GRPC_PORT="${SGLANG_DECODE_GRPC_PORT:-30011}"
SGLANG_PREFILL_GPU="${SGLANG_PREFILL_GPU:-0}"
SGLANG_DECODE_GPU="${SGLANG_DECODE_GPU:-1}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner "Launching SGLang Native-gRPC Disaggregated Serving (2 GPUs)" "$MODEL" "$HTTP_PORT"

python3 -m dynamo.frontend &

CUDA_VISIBLE_DEVICES="$SGLANG_PREFILL_GPU" \
    "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PREFILL_HTTP_PORT" \
    --grpc-port "$SGLANG_PREFILL_GRPC_PORT" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$SGLANG_DISAGGREGATION_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

CUDA_VISIBLE_DEVICES="$SGLANG_DECODE_GPU" \
    "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_DECODE_HTTP_PORT" \
    --grpc-port "$SGLANG_DECODE_GRPC_PORT" \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$SGLANG_DISAGGREGATION_BOOTSTRAP_PORT" \
    --disaggregation-transfer-backend nixl \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-prefill \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-sglang-remote \
    --sglang-endpoint "${SGLANG_HOST}:${SGLANG_PREFILL_GRPC_PORT}" \
    --bootstrap-host "$SGLANG_BOOTSTRAP_HOST" &

OTEL_SERVICE_NAME=dynamo-worker-decode \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-sglang-remote \
    --sglang-endpoint "${SGLANG_HOST}:${SGLANG_DECODE_GRPC_PORT}" &

wait_any_exit
