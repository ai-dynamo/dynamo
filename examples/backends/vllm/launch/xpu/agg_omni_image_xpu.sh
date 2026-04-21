#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export VLLM_OMNI_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="Qwen/Qwen-Image"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching vLLM-Omni Image Generation (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s -X POST http://localhost:${HTTP_PORT}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "A red apple on a white table",
    "size": "512x512",
    "num_inference_steps": 20
  }' | jq
CURL


python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

# Conservative defaults for single-XPU diffusion model loading. These can still
# be overridden by explicit CLI args in EXTRA_ARGS.
DIFFUSION_SAFE_ARGS=(
  --enable-layerwise-offload
  --layerwise-num-gpu-layers "${LAYERWISE_NUM_GPU_LAYERS:-1}"
  --enforce-eager
)

echo "Starting Omni worker..."
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities image \
    --media-output-fs-url file:///tmp/dynamo_media \
  "${DIFFUSION_SAFE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
