#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

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
print_launch_banner --no-curl "Launching vLLM-Omni Audio/TTS (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
  curl -X POST http://localhost:${HTTP_PORT}/v1/audio/speech \\
    -H 'Content-Type: application/json' \\
    -d '{
      "input": "Hey, this is generated using Dynamo!",
      "model": "${MODEL}",
      "voice": "vivian",
      "language": "English"
    }' \\
    -o dynamo-audio.wav
CURL


python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni Audio worker..."
# Upstream qwen3_tts stage configs still use a 65536 stage-1 max_model_len.
# vLLM 0.19 validates that against the model config unless we opt in here.
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    --enforce-eager \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
