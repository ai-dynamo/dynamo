#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch an aggregated vLLM-Omni deployment for image-to-video (I2V).
#
# Usage:
#   bash agg_omni_i2v.sh [OPTIONS]
#
# Options:
#   --model <model>   Model to use (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers)
#   Any other flags are forwarded to the vLLM worker.
set -e
trap 'echo Cleaning up...; kill 0' EXIT


MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --model requires a value" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "Starting vLLM-Omni I2V Worker"
echo "Model: $MODEL"
echo "=========================================="


echo "Starting frontend on port ${DYN_HTTP_PORT:-8000}..."
python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --output-modalities video \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}"
