#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated Cosmos3 image-to-video generation (1 GPU).
# Same worker as text-to-video (registers the "video" modality); i2v is driven
# by adding "input_reference" to the /v1/videos request. The image loader
# rejects local file paths — pass a data: URI (base64) or an http(s) URL.
# --no-cosmos3-guardrails skips loading the safety guardrail models.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="nvidia/Cosmos3-Nano"

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
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
print_launch_banner --no-curl "Launching vLLM-Omni Cosmos3 Image-to-Video (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
# Official Cosmos3 image-to-video payload (prompt + vision_path verbatim).
# input_reference must be an http(s) URL or a data: URI (local paths are rejected).
curl -s http://localhost:${HTTP_PORT}/v1/videos \\
  -H 'Content-Type: application/json' \\
  --data-binary @${SCRIPT_DIR}/cosmos3/i2v.json | jq
CURL


python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities video \
    --no-cosmos3-guardrails \
    --media-output-fs-url file:///tmp/dynamo_media \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
