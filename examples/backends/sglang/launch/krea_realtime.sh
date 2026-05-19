#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Krea realtime chunked video generation (sgl-project/sglang#19817).
# Streams MP4 chunks out as SSE events via POST /v1/videos with stream:true.
# GPUs: 1 (H200/B200 recommended per the PR's perf table).

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Defaults — mirror the antgroup demo (832x480, fps=12, num_inference_steps=4)
MODEL_PATH="Wan-AI/krea-realtime-video-diffusers"
FS_URL="file:///tmp/dynamo_media"
HTTP_PORT="${DYN_HTTP_PORT:-${HTTP_PORT:-8000}}"
WIDTH=832
HEIGHT=480
FPS=12
NUM_FRAMES=36
NUM_INFERENCE_STEPS=4
SECONDS_DEFAULT=3

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)        MODEL_PATH="$2"; shift 2 ;;
        --fs-url)            FS_URL="$2"; shift 2 ;;
        --http-port)         HTTP_PORT="$2"; shift 2 ;;
        --width)             WIDTH="$2"; shift 2 ;;
        --height)            HEIGHT="$2"; shift 2 ;;
        --fps)               FPS="$2"; shift 2 ;;
        --num-frames)        NUM_FRAMES="$2"; shift 2 ;;
        --num-inference-steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS]

Launch a Dynamo Krea realtime video worker. MP4 chunks stream out as SSE
events via POST /v1/videos with stream:true.

Options:
  --model-path <path>          Diffusers model id (default: $MODEL_PATH)
  --fs-url <url>                Filesystem URL for media storage (default: $FS_URL)
  --http-port <port>            Frontend HTTP port (default: $HTTP_PORT)
  --width <n>                   Video width  (default: $WIDTH)
  --height <n>                  Video height (default: $HEIGHT)
  --fps <n>                     Frames per second (default: $FPS)
  --num-frames <n>              Default frame count for sample curl (default: $NUM_FRAMES)
  --num-inference-steps <n>     Denoising steps per chunk (default: $NUM_INFERENCE_STEPS)
  -h, --help                    Show this help

Additional flags are forwarded to dynamo.sglang.
USAGE
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

print_launch_banner --no-curl "Launching Krea Realtime Video Worker" "$MODEL_PATH" "$HTTP_PORT" \
    "FS URL:      $FS_URL" \
    "Resolution:  ${WIDTH}x${HEIGHT}" \
    "FPS:         $FPS" \
    "Steps/chunk: $NUM_INFERENCE_STEPS"

print_curl_footer <<CURL
  # Inline MP4 chunks over SSE (data[0].b64_json on each event):
  curl -N http://localhost:${HTTP_PORT}/v1/videos \\
    -H 'Content-Type: application/json' \\
    -H 'Accept: text/event-stream' \\
    -d '{
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "model": "${MODEL_PATH}",
      "seconds": ${SECONDS_DEFAULT},
      "size": "${WIDTH}x${HEIGHT}",
      "stream": true,
      "response_format": "b64_json",
      "nvext": {
        "fps": ${FPS},
        "num_frames": ${NUM_FRAMES},
        "num_inference_steps": ${NUM_INFERENCE_STEPS}
      }
    }'

  # URL mode: chunks uploaded to ${FS_URL} as <request_id>_<NNNN>.mp4
  # (data[0].url on each event):
  curl -N http://localhost:${HTTP_PORT}/v1/videos \\
    -H 'Content-Type: application/json' \\
    -H 'Accept: text/event-stream' \\
    -d '{
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "model": "${MODEL_PATH}",
      "seconds": ${SECONDS_DEFAULT},
      "size": "${WIDTH}x${HEIGHT}",
      "stream": true,
      "response_format": "url",
      "nvext": {
        "fps": ${FPS},
        "num_frames": ${NUM_FRAMES},
        "num_inference_steps": ${NUM_INFERENCE_STEPS}
      }
    }'
CURL

echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python3 -m dynamo.frontend --http-port "$HTTP_PORT" &

sleep 2

echo "Starting Krea Realtime Video Worker..."
python3 -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_PATH" \
    --tp 1 \
    --realtime-video-worker \
    --media-output-fs-url "$FS_URL" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --enable-metrics \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
