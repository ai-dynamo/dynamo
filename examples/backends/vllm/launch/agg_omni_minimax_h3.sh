#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch one aggregated vLLM-Omni worker that serves MiniMax-H3 T2VA.
set -euo pipefail
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# MiniMax-H3 uses the four visible B200s for diffusion parallelism. The
# text-generation KV-cache flags from gpu_utils.sh do not apply to this worker.

MODEL="${DYN_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
MODEL_PATH="${DYN_H3_MODEL_PATH:-}"
MODEL_REVISION="${DYN_H3_MODEL_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
ULYSSES_DEGREE="${DYN_H3_ULYSSES_DEGREE:-4}"
TEXT_ENCODER_TP_SIZE="${DYN_H3_TEXT_ENCODER_TP_SIZE:-4}"
# H3's native patch-parallel VAE requires at least one spatial tile per rank.
# Keep it single-rank by default so small supported resolutions (for example,
# 448x256) do not leave ranks with empty tile lists. Larger workloads may opt
# into the full DiT group size explicitly.
VAE_PATCH_PARALLEL_SIZE="${DYN_H3_VAE_PATCH_PARALLEL_SIZE:-1}"
ATTENTION_BACKEND="${DYN_H3_ATTENTION_BACKEND:-TRTLLM_ATTN}"
FASTH3_LORA_PATH="${DYN_H3_FASTH3_LORA_PATH:-}"
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
        --model-path)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --model-path requires a value" >&2
                exit 1
            fi
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-revision)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --model-revision requires a value" >&2
                exit 1
            fi
            MODEL_REVISION="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

MODEL_PATH="${MODEL_PATH:-$MODEL}"
FASTH3_ARGS=()
if [[ -n "$FASTH3_LORA_PATH" ]]; then
    if [[ ! -f "$FASTH3_LORA_PATH" ]]; then
        echo "FastH3 adapter not found: $FASTH3_LORA_PATH" >&2
        exit 1
    fi
    FASTH3_ARGS=(--lora-path "$FASTH3_LORA_PATH")
    EXAMPLE_INFERENCE_STEPS=4
    EXAMPLE_SCHEDULER_FIELDS=""
else
    EXAMPLE_INFERENCE_STEPS=50
    EXAMPLE_SCHEDULER_FIELDS=$',\n    "flow_shift": 12.0,\n    "audio_flow_shift": 3.0'
fi

SERVED_MODEL_ARGS=()
if [[ "$MODEL_PATH" != "$MODEL" ]]; then
    SERVED_MODEL_ARGS=(--served-model-name "$MODEL")
fi

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E '(^| )libx264( |$)' >/dev/null
python -c 'import av; av.codec.Codec("h264", "w"); av.codec.Codec("aac", "w")'

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-3600}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching MiniMax-H3 T2VA (4 GPUs)" "$MODEL_PATH" "$HTTP_PORT"
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
      "fps": 24,
      "num_inference_steps": ${EXAMPLE_INFERENCE_STEPS},
      "seed": 42
    },
    "task": "t2va",
    "duration": 10.0,
    "aspect_ratio": "16:9"${EXAMPLE_SCHEDULER_FIELDS}
  }' | jq
CURL

python -m dynamo.frontend &

if ! wait_for_ready "http://localhost:${HTTP_PORT}/health" 60; then
    echo "Frontend failed to become ready; stopping launch." >&2
    exit 1
fi

echo "Starting MiniMax-H3 T2VA Omni worker..."
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python -m dynamo.vllm.omni \
    --model "$MODEL_PATH" \
    "${SERVED_MODEL_ARGS[@]}" \
    --revision "$MODEL_REVISION" \
    --output-modalities video \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    --task-type fl2va \
    --ulysses-degree "$ULYSSES_DEGREE" \
    --text-encoder-tp-size "$TEXT_ENCODER_TP_SIZE" \
    --vae-patch-parallel-size "$VAE_PATCH_PARALLEL_SIZE" \
    --vae-use-tiling \
    --diffusion-attention-backend "$ATTENTION_BACKEND" \
    "${FASTH3_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
