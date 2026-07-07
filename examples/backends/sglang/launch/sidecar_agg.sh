#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving through SGLang's native gRPC server (1 GPU).
# Requires an SGLang build containing PR #23508 and, until it merges, PR #25185.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-path <name>] [SGLang options...]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-30001}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

print_launch_banner "Launching SGLang Native-gRPC Sidecar (1 GPU)" "$MODEL" "$HTTP_PORT"

python3 -m dynamo.frontend &

# --grpc-port enables the native Rust gRPC server alongside SGLang's HTTP API.
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_HTTP_PORT" \
    --grpc-port "$SGLANG_GRPC_PORT" \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-sglang-sidecar \
    --sglang-endpoint "${SGLANG_HOST}:${SGLANG_GRPC_PORT}" &

wait_any_exit
