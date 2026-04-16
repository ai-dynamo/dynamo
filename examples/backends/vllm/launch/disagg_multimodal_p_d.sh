#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated multimodal P/D serving (no separate encode worker).
# Prefill handles image loading via PIL and computes image_grid_thw
# for the decode worker using Qwen2VLImageProcessor's smart_resize.
#
# This is a simpler deployment than E/P/D: only 2 workers instead of 3.
# Trade-off: prefill does vision encoding internally (no dedicated encoder),
# which uses more GPU memory on the prefill worker.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
SINGLE_GPU=false

# Parse command line arguments
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
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with Prefill/Decode workers (no encoder)"
            echo "Prefill loads images via PIL and computes grid metadata for decode."
            echo ""
            echo "Options:"
            echo "  --model <model_name>   Specify the VLM model (default: $MODEL_NAME)"
            echo "  --single-gpu           Pack both workers on 1 GPU (for small models)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_LABEL="1 GPU"
else
    GPU_LABEL="2 GPUs"
fi
print_launch_banner --multimodal "Launching Disaggregated Multimodal P/D ($GPU_LABEL)" "$MODEL_NAME" "$HTTP_PORT"

# Start frontend
echo "Starting frontend..."
python -m dynamo.frontend &

EXTRA_ARGS=""
PD_EXTRA_ARGS=""

# GPU assignments
DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-0}
DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-1}

# GPU memory utilization
DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}

PD_KV_CACHE_BYTES=$((512 * 1024 * 1024))

if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_PREFILL_WORKER_GPU=0
    DYN_DECODE_WORKER_GPU=0
    DYN_PREFILL_GPU_MEM=0.45
    DYN_DECODE_GPU_MEM=0.45
    EXTRA_ARGS="--enforce-eager"
    PD_EXTRA_ARGS="--max-model-len 4096 \
--kv-cache-memory-bytes $PD_KV_CACHE_BYTES \
--limit-mm-per-prompt {\"image\":1,\"video\":0,\"audio\":0}"
fi

# Start prefill worker (handles image loading internally, no --route-to-encoder)
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU (GPU mem: $DYN_PREFILL_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU \
python -m dynamo.vllm \
  --disaggregation-mode prefill \
  --enable-multimodal \
  --model $MODEL_NAME \
  --gpu-memory-utilization $DYN_PREFILL_GPU_MEM \
  $EXTRA_ARGS \
  $PD_EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

# Start decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU (GPU mem: $DYN_DECODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU \
python -m dynamo.vllm \
  --disaggregation-mode decode \
  --enable-multimodal \
  --enable-mm-embeds \
  --model $MODEL_NAME \
  --gpu-memory-utilization $DYN_DECODE_GPU_MEM \
  $EXTRA_ARGS \
  $PD_EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
