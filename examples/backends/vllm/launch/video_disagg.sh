#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated video serving with standard Dynamo preprocessing and vLLM backend.

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
SINGLE_GPU=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS]

Options:
  --model <model_name>   Video-capable VLM to serve (default: $MODEL_NAME)
  --single-gpu           Run prefill and decode on one GPU for functional testing
  -h, --help             Show this help message

Any arguments after '--' are passed through to both vLLM workers.
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

if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_LABEL="1 GPU"
    PREFILL_GPU="${DYN_PREFILL_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
    DECODE_GPU="${DYN_DECODE_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    PD_KV_CACHE_BYTES=$((512 * 1024 * 1024))
    SHARED_GPU_FRACTION=$(build_gpu_mem_args vllm --workers-per-gpu 2)
    PREFILL_GPU_MEM="${DYN_PREFILL_GPU_MEM:-${SHARED_GPU_FRACTION:-0.45}}"
    DECODE_GPU_MEM="${DYN_DECODE_GPU_MEM:-${SHARED_GPU_FRACTION:-0.45}}"
    SHARED_ARGS=(
        --enforce-eager
        --max-model-len "$MAX_MODEL_LEN"
        --kv-cache-memory-bytes "$PD_KV_CACHE_BYTES"
        --limit-mm-per-prompt '{"image":1,"video":1,"audio":0}'
    )
else
    GPU_LABEL="2 GPUs"
    PREFILL_GPU="${DYN_PREFILL_WORKER_GPU:-0}"
    DECODE_GPU="${DYN_DECODE_WORKER_GPU:-1}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    GPU_MEM_ARGS=$(build_gpu_mem_args vllm)
    PREFILL_GPU_MEM="${DYN_PREFILL_GPU_MEM:-${GPU_MEM_ARGS:-0.9}}"
    DECODE_GPU_MEM="${DYN_DECODE_GPU_MEM:-${GPU_MEM_ARGS:-0.9}}"
    SHARED_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

print_launch_banner --no-curl "Launching Disaggregated Video Serving ($GPU_LABEL)" "$MODEL_NAME" "$HTTP_PORT" \
    "Backend:     Prefill + decode workers via dynamo.vllm" \
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

VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
    python -m dynamo.vllm \
        --disaggregation-mode prefill \
        --enable-multimodal \
        --model "$MODEL_NAME" \
        --gpu-memory-utilization "$PREFILL_GPU_MEM" \
        "${SHARED_ARGS[@]}" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' \
        "${EXTRA_ARGS[@]}" &

VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
    python -m dynamo.vllm \
        --disaggregation-mode decode \
        --enable-multimodal \
        --model "$MODEL_NAME" \
        --gpu-memory-utilization "$DECODE_GPU_MEM" \
        "${SHARED_ARGS[@]}" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' \
        "${EXTRA_ARGS[@]}" &

wait_any_exit
