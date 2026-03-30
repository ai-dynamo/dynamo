#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

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
print_launch_banner --no-curl "Launching vLLM-Omni Video Generation (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "Dog running on a beach",
    "size": "832x480",
    "response_format": "url",
    "nvext": {
      "num_inference_steps": 20,
      "num_frames": 30
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
