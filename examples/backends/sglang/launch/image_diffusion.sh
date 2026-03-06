#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Image diffusion worker (text-to-image). Default model: FLUX.1-dev (~38 GB VRAM).
# GPUs: 1

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

set -e

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $FRONTEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Defaults
MODEL_PATH="black-forest-labs/FLUX.1-dev"
FS_URL="file:///tmp/dynamo_media"
HTTP_URL=""
HTTP_PORT="${HTTP_PORT:-8000}"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --fs-url)
            FS_URL="$2"
            shift 2
            ;;
        --http-url)
            HTTP_URL="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Launch a Dynamo image diffusion worker."
            echo ""
            echo "Options:"
            echo "  --model-path <path>          Model path (default: black-forest-labs/FLUX.1-dev)"
            echo "  --fs-url <url>               Filesystem URL for image storage (default: file:///tmp/dynamo_media)"
            echo "  --http-url <url>             Base URL for serving images over HTTP (optional)"
            echo "  --http-port <port>           Frontend HTTP port (default: 8000)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Additional flags are forwarded to dynamo.sglang."
            echo ""
            echo "Examples:"
            echo "  # Local file storage"
            echo "  $0 --model-path black-forest-labs/FLUX.1-dev --fs-url file:///tmp/images"
            echo ""
            echo "  # S3 storage (set FSSPEC_S3_KEY, FSSPEC_S3_SECRET, optionally FSSPEC_S3_ENDPOINT_URL)"
            echo "  $0 --fs-url s3://my-bucket/images"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

EXTRA_INFO=("FS URL:      $FS_URL")
[ -n "$HTTP_URL" ] && EXTRA_INFO+=("HTTP URL:    $HTTP_URL")
print_launch_banner --no-curl "Launching Image Diffusion Worker" "$MODEL_PATH" "$HTTP_PORT" \
    "${EXTRA_INFO[@]}"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/images/generations \\
    -H 'Content-Type: application/json' \\
    -d '{
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "model": "${MODEL_PATH}",
      "size": "1024x1024",
      "response_format": "url",
      "nvext": {
        "num_inference_steps": 15
      }
    }'
CURL

# Build optional HTTP URL arg
HTTP_URL_ARGS=()
if [ -n "$HTTP_URL" ]; then
    HTTP_URL_ARGS=(--media-output-http-url "$HTTP_URL")
fi

# Launch frontend
echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" &
FRONTEND_PID=$!

sleep 2

# Launch image diffusion worker
echo "Starting Image Diffusion Worker..."
python3 -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_PATH" \
    --image-diffusion-worker \
    --media-output-fs-url "$FS_URL" \
    "${HTTP_URL_ARGS[@]}" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --enable-metrics \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
