#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch one aggregated vLLM-Omni worker that serves MiniMax-H3 T2VA.
set -euo pipefail
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${DYN_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
ULYSSES_DEGREE="${DYN_H3_ULYSSES_DEGREE:-4}"
TEXT_ENCODER_TP_SIZE="${DYN_H3_TEXT_ENCODER_TP_SIZE:-4}"
# H3's native patch-parallel VAE requires at least one spatial tile per rank.
# Keep it single-rank by default so small supported resolutions (for example,
# 448x256) do not leave ranks with empty tile lists. Larger workloads may opt
# into the full DiT group size explicitly.
VAE_PATCH_PARALLEL_SIZE="${DYN_H3_VAE_PATCH_PARALLEL_SIZE:-1}"
ATTENTION_BACKEND="${DYN_H3_ATTENTION_BACKEND:-TRTLLM_ATTN}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
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

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264rgb >/dev/null
python -c 'import av; av.codec.Codec("h264", "w"); av.codec.Codec("aac", "w")'

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-3600}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching MiniMax-H3 T2VA (4 GPUs)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -sS http://localhost:${HTTP_PORT}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "A cat playing Canon in D on a grand piano.",
    "size": "448x256",
    "response_format": "url",
    "output_format": "mp4",
    "nvext": {
      "num_inference_steps": 50,
      "seed": 42
    },
    "task": "t2va",
    "duration": 10.0,
    "aspect_ratio": "16:9",
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0
  }' | jq
CURL

python -m dynamo.frontend &

sleep 2

echo "Starting MiniMax-H3 T2VA Omni worker..."
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities video \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    --task-type t2va \
    --ulysses-degree "$ULYSSES_DEGREE" \
    --text-encoder-tp-size "$TEXT_ENCODER_TP_SIZE" \
    --vae-patch-parallel-size "$VAE_PATCH_PARALLEL_SIZE" \
    --vae-use-tiling \
    --diffusion-attention-backend "$ATTENTION_BACKEND" \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
