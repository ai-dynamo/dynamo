#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated video serving with standard Dynamo preprocessing and vLLM backend.

set -euo pipefail

cleanup() {
    echo "Cleaning up..."
    local pids
    pids="$(jobs -pr)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
    fi
}

trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

export PYTHONPATH="${REPO_ROOT}/components/src:${REPO_ROOT}/lib/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS]

Options:
  --model <model_name>   Video-capable VLM to serve (default: $MODEL_NAME)
  -h, --help             Show this help message

Any arguments after '--' are passed through to the vLLM worker.
USAGE
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

export DYN_REQUEST_PLANE=tcp

GPU_MEM_ARGS=$(build_gpu_mem_args vllm)

print_launch_banner --no-curl "Launching Aggregated Video Serving" "$MODEL_NAME" "$HTTP_PORT" \
    "Backend:     dynamo.vllm --enable-multimodal" \
    "Video path:  Standard TokensPrompt multi_modal_data flow"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL_NAME}",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the video in detail"},
        {"type": "video_url", "video_url": {"url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"}}
      ]}],
      "max_tokens": 128
    }'
CURL

python -m dynamo.frontend &

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    python -m dynamo.vllm \
        --enable-multimodal \
        --model "$MODEL_NAME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        $GPU_MEM_ARGS \
        "${EXTRA_ARGS[@]}" &

wait_any_exit
