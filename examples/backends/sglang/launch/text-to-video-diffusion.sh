#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Text-to-video generation with Wan2.1 models.
# GPUs: 1 (--wan-size 1b) or 2 (--wan-size 14b)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Defaults
WAN_SIZE="1b"
FS_URL="file:///tmp/dynamo_media"
HTTP_PORT="${HTTP_PORT:-8000}"
NUM_FRAMES=17
HEIGHT=480
WIDTH=832
NUM_INFERENCE_STEPS=50

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --wan-size)
            WAN_SIZE="$2"
            shift 2
            ;;
        --fs-url)
            FS_URL="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --num-inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Launch a Dynamo T2V (text-to-video) worker with Wan models."
            echo ""
            echo "Options:"
            echo "  --wan-size <1b|14b>          Model size (default: 1b)"
            echo "  --fs-url <url>               Filesystem URL for media storage (default: file:///tmp/dynamo_media)"
            echo "  --http-port <port>            Frontend HTTP port (default: 8000)"
            echo "  --num-frames <n>              Default frame count for health check (default: 17)"
            echo "  --height <n>                  Video height (default: 480)"
            echo "  --width <n>                   Video width (default: 832)"
            echo "  --num-inference-steps <n>     Denoising steps (default: 50)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Additional flags are forwarded to dynamo.sglang."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Select model and TP based on size
case "$WAN_SIZE" in
    1b|1B)
        MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        TP_SIZE=1
        ;;
    14b|14B)
        MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
        TP_SIZE=2
        ;;
    *)
        echo "Error: --wan-size must be '1b' or '14b', got '$WAN_SIZE'"
        exit 1
        ;;
esac

print_launch_banner --no-curl "Launching T2V Video Generation Worker" "$MODEL_PATH" "$HTTP_PORT" \
    "TP Size:     $TP_SIZE" \
    "FS URL:      $FS_URL" \
    "Resolution:  ${WIDTH}x${HEIGHT}"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/videos \\
    -H 'Content-Type: application/json' \\
    -d '{
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "model": "${MODEL_PATH}",
      "seconds": 2,
      "size": "${WIDTH}x${HEIGHT}",
      "response_format": "url",
      "nvext": {
        "fps": 8,
        "num_frames": ${NUM_FRAMES},
        "num_inference_steps": ${NUM_INFERENCE_STEPS}
      }
    }'
CURL

# Launch frontend
echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" &

sleep 2

# Launch video generation worker
echo "Starting T2V Worker ($WAN_SIZE)..."
python3 -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_PATH" \
    --tp "$TP_SIZE" \
    --video-generation-worker \
    --media-output-fs-url "$FS_URL" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --enable-metrics \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
