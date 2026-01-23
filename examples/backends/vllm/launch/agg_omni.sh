#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for aggregated Omni worker (text-to-image POC)
# Usage: ./agg_omni.sh [OPTIONS]
#   --model <model_name>    Model to use (default: Qwen/Qwen-Image)
#   --http-port <port>      HTTP port for frontend (default: 8000)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL="Qwen/Qwen-Image"
HTTP_PORT=8000

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name>     Specify the Omni model to use (default: $MODEL)"
            echo "  --http-port <port>       HTTP port for frontend (default: $HTTP_PORT)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=== Launching Aggregated Omni Worker ==="
echo "Model: $MODEL"
echo "HTTP Port: $HTTP_PORT"
echo "========================================="

# Launch frontend (Rust HTTP server)
python -m dynamo.frontend --http-port $HTTP_PORT &

echo "Frontend started on port $HTTP_PORT"
echo "Starting Omni backend worker..."

# Launch Omni backend worker (runs in foreground)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.main \
    --model "$MODEL" \
    --omni-worker \
    --served-model-name "$(basename "$MODEL")" \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --tensor-parallel-size 1 \
    --connector none \
    "${EXTRA_ARGS[@]}"
