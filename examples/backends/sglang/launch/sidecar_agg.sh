#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the OpenEngine sidecar (1 GPU).
#
# Unlike agg.sh (which runs SGLang in-process inside the Dynamo worker), this
# launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. A native SGLang engine exposing the OpenEngine v1 gRPC server
#      (`sglang.launch_server --openengine-port`)
#   3. The Dynamo SGLang sidecar worker, which talks to (2) over OpenEngine
#
# The SGLang engine mounts the OpenEngine service (Rust, in sglang.srt.grpc._core)
# bridged to its scheduler. The Dynamo sidecar (dynamo-sglang-sidecar) never
# imports sglang; it is given ONLY the OpenEngine endpoint and discovers the
# model, role, parallelism, and KV config from the engine over OpenEngine RPCs.

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
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional options are passed through to sglang.launch_server."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
OPENENGINE_HOST="${OPENENGINE_HOST:-127.0.0.1}"
OPENENGINE_PORT="${OPENENGINE_PORT:-50051}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching SGLang Sidecar Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python3 -m dynamo.frontend &

# 2. Native SGLang engine with the OpenEngine gRPC server. Setting
# --openengine-port selects the OpenEngine serve path instead of the SGLang
# HTTP/native-gRPC server. Engine role/capabilities are discovered, not flagged.
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$OPENENGINE_HOST" \
    --openengine-host "$OPENENGINE_HOST" \
    --openengine-port "$OPENENGINE_PORT" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no sglang import; OpenEngine client only). It
# receives ONLY the endpoint; model and role are discovered from the engine.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-sglang-sidecar \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit