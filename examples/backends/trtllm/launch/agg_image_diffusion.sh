#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated image diffusion serving with TensorRT-LLM backend.
# Uses FLUX.1-dev by default (1 GPU).

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_trtllm_override_args_with_mem

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"black-forest-labs/FLUX.1-dev"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"black-forest-labs/FLUX.1-dev"}
export MEDIA_OUTPUT_FS_URL=${MEDIA_OUTPUT_FS_URL:-"file:///tmp/dynamo_media"}

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.trtllm."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build GPU memory JSON (returns bare JSON, no flag)
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)

# Add --override-engine-args if we have JSON
TRTLLM_OVERRIDE_ARGS=()
if [[ -n "$OVERRIDE_JSON" ]]; then
    TRTLLM_OVERRIDE_ARGS=(--override-engine-args "$OVERRIDE_JSON")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching Image Diffusion Serving (1 GPU)" "$MODEL_PATH" "$HTTP_PORT" \
    "Media URL:   $MEDIA_OUTPUT_FS_URL"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/images/generations \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${SERVED_MODEL_NAME}",
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "size": "256x256",
      "nvext": {"num_inference_steps": 10, "seed": 42}
    }'
CURL

# run frontend
python3 -m dynamo.frontend &

# run image diffusion worker
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --modality image_diffusion \
  --media-output-fs-url "$MEDIA_OUTPUT_FS_URL" \
  "${TRTLLM_OVERRIDE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
