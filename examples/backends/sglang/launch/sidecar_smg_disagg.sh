#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving via the upstream SGLang SMG sidecar.
# GPUs: 2 (prefill on GPU 0, decode on GPU 1).
#
# This launches five processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. SGLang prefill engine with --grpc-mode
#   3. Dynamo SGLang SMG sidecar registered as prefill
#   4. SGLang decode engine with --grpc-mode
#   5. Dynamo SGLang SMG sidecar registered as decode/backend
#
# The SGLang environment must include the optional gRPC dependencies:
#   pip install 'smg-grpc-servicer[sglang]>=0.5.2'

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
            echo "Additional options are passed through to both SGLang engine processes."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SMG_CONNECT_HOST="${SMG_CONNECT_HOST:-127.0.0.1}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-30001}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-30002}"
PREFILL_SMG_PORT="${PREFILL_SMG_PORT:-40001}"
DECODE_SMG_PORT="${DECODE_SMG_PORT:-40002}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"
SMG_CONNECTIONS="${SMG_CONNECTIONS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching SGLang SMG Sidecar Disaggregated Serving (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Prefill SMG: ${SMG_CONNECT_HOST}:${PREFILL_SMG_PORT}" \
    "Decode SMG:  ${SMG_CONNECT_HOST}:${DECODE_SMG_PORT}" \
    "Bootstrap:   ${DISAGG_BOOTSTRAP_PORT}"

"$PYTHON_BIN" -m dynamo.frontend &

CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
SGLANG_GRPC_PORT="$PREFILL_SMG_PORT" \
"$PYTHON_BIN" -m sglang.launch_server \
    --grpc-mode \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$PREFILL_HTTP_PORT" \
    --tp 1 \
    --trust-remote-code \
    --disable-cuda-graph \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-prefill \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
    dynamo-sglang-smg-sidecar \
    --smg-endpoint "${SMG_CONNECT_HOST}:${PREFILL_SMG_PORT}" \
    --smg-connections "$SMG_CONNECTIONS" &

CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
SGLANG_GRPC_PORT="$DECODE_SMG_PORT" \
"$PYTHON_BIN" -m sglang.launch_server \
    --grpc-mode \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$DECODE_HTTP_PORT" \
    --tp 1 \
    --trust-remote-code \
    --disable-cuda-graph \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-decode \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
    dynamo-sglang-smg-sidecar \
    --smg-endpoint "${SMG_CONNECT_HOST}:${DECODE_SMG_PORT}" \
    --smg-connections "$SMG_CONNECTIONS" &

wait_any_exit
