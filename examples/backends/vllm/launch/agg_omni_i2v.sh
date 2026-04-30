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

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# Not a valid PNG, example only
INPUT_REFERENCE_DATA_URL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX3kAAAAASUVORK5CYII="

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

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching vLLM-Omni Image-to-Video (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "A bear sleeping",
    "input_reference": "${INPUT_REFERENCE_DATA_URL}",
    "size": "832x480",
    "response_format": "url",
    "nvext": {
      "num_inference_steps": 40,
      "num_frames": 33,
      "guidance_scale": 1.0,
      "boundary_ratio": 0.875
    }
  }' | jq
CURL

python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities video \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
