#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aggregated correctness topology: a custom Qwen2.5-VL vision producer feeds
# vLLM's native Qwen2.5-VL external-multimodal input path.

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../common/launch_utils.sh"

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen2_5_vl_native_encoder.Qwen2_5VLNativeEncoder}"
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${DYN_MAX_NUM_SEQS:-8}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL=$2; shift 2 ;;
        --encoder-class)
            ENCODER_CLASS=$2; shift 2 ;;
        --gpu)
            WORKER_GPU=$2; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: agg_qwen2_5_vl_native.sh [OPTIONS]

Run a custom Qwen2.5-VL encoder with a native Qwen2.5-VL downstream model.

Options:
  --model <id>           Qwen2/2.5-VL checkpoint
  --encoder-class <path> Dotted native-output CustomEncoder class
  --gpu <index>          GPU index for the aggregated worker
  -h, --help             Show this help

Environment variables:
  DYN_MODEL                          Downstream Qwen2/2.5-VL checkpoint
  DYN_ENCODER_CLASS                  Dotted CustomEncoder class
  DYN_QWEN2_VL_ENCODER_MODEL         Vision-tower checkpoint
  DYN_QWEN2_VL_DISABLE_CUDA_GRAPHS   Defaults to 1 for correctness coverage
EOF
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

export DYN_QWEN2_VL_ENCODER_MODEL="${DYN_QWEN2_VL_ENCODER_MODEL:-$MODEL}"
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE="${DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE:-2048}"
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY="${DYN_QWEN2_VL_PREPROCESS_CONCURRENCY:-8}"
export DYN_QWEN2_VL_MAX_BATCH_COST="${DYN_QWEN2_VL_MAX_BATCH_COST:-8}"
export DYN_QWEN2_VL_DISABLE_CUDA_GRAPHS="${DYN_QWEN2_VL_DISABLE_CUDA_GRAPHS:-1}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -z "$GPU_MEM_ARGS" ]] && GPU_MEM_ARGS="--gpu-memory-utilization ${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.55}"

print_launch_banner --multimodal "Qwen2.5-VL CustomEncoder — Native Aggregated" \
    "$MODEL" "$HTTP_PORT" \
    "Worker GPU:  $WORKER_GPU" \
    "Encoder:     $ENCODER_CLASS" \
    "Input route: vLLM native external multimodal embeddings"

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

echo "[1/2] Starting frontend (port $HTTP_PORT)..."
python -m dynamo.frontend &

echo "[2/2] Starting native Qwen worker (model=$MODEL, GPU=$WORKER_GPU)..."
CUDA_VISIBLE_DEVICES=$WORKER_GPU \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python -m dynamo.vllm \
    --model "$MODEL" \
    --custom-encoder-class "$ENCODER_CLASS" \
    --enable-multimodal \
    --enable-mm-embeds \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
