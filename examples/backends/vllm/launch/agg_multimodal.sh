#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal image/video serving with standard Dynamo preprocessing
#
# Architecture: Single-worker PD (Prefill-Decode)
# - Frontend: Rust OpenAIPreprocessor forwards multimodal requests
# - Worker: Standard vLLM worker with multimodal model support
#
# For EPD (Encode-Prefill-Decode) architecture with dedicated encoding worker,
# see agg_multimodal_epd.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Instruct-FP8}"

# Parse command line arguments
# Extra arguments are passed through to the vLLM worker
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS]"
            echo "Options:"
            echo "  --model <model_name>   Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Any additional arguments are passed through to the vLLM worker."
            echo "Example: $0 --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --dyn-tool-call-parser hermes"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Use TCP transport (instead of default NATS)
# TCP is preferred for multimodal workloads because it overcomes:
# - NATS default 1MB max payload limit (multimodal base64 images can exceed this)
export DYN_REQUEST_PLANE=tcp

print_launch_banner --no-curl "Launching Aggregated Multimodal Serving" "$MODEL_NAME" "$HTTP_PORT" \
    "Backend:     dynamo.vllm --enable-multimodal" \
    "Media:       image_url and video_url (model support dependent)"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL_NAME}",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}
      ]}],
      "max_tokens": 50
    }'

  # For video-capable models such as Qwen/Qwen3-VL-2B-Instruct:
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "Qwen/Qwen3-VL-2B-Instruct",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the video in detail"},
        {"type": "video_url", "video_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"}}
      ]}],
      "max_tokens": 128
    }'
CURL

# Start frontend with Rust OpenAIPreprocessor
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# ---- Per-model defaults ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MODEL_EXTRA_ARGS=""
case "$MODEL_NAME" in
    meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
        MAX_MODEL_LEN="108960"
        MODEL_EXTRA_ARGS="--tensor-parallel-size=8" ;;
esac

# Default KV cache cap from profiling (2x safety over min=461 MiB); ~9.6 GiB peak VRAM
# Uses smallest profiled value across multimodal tests; profiler/test framework overrides via env
: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=922354000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

# Start vLLM worker with vision model
# --enforce-eager: Quick deployment (remove for production)
# Extra args from command line come last to allow overrides
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --enable-multimodal --model $MODEL_NAME \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    $MODEL_EXTRA_ARGS \
    "${EXTRA_ARGS[@]}"

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
