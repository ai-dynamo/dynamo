#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dual Encoder E/P/D configuration
# 2 Encoder workers (1 CPU + 1 XPU) + Prefill (XPU) + Decode (XPU)
# Load balancing across encoders via round-robin

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Dual encoder disaggregated multimodal serving"
            echo "  - 2 Encoder workers (1 on CPU, 1 on XPU)"
            echo "  - Prefill worker (XPU)"
            echo "  - Decode worker (XPU)"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model Qwen/Qwen2.5-VL-3B-Instruct"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Device platform and affinity env name
DEVICE_PLATFORM="${DEVICE_PLATFORM:-cuda}"
if [[ -z "${DEVICE_AFFINITY_ENV:-}" ]]; then
    if [[ "${DEVICE_PLATFORM,,}" == "xpu" ]]; then
        DEVICE_AFFINITY_ENV="ZE_AFFINITY_MASK"
    else
        DEVICE_AFFINITY_ENV="CUDA_VISIBLE_DEVICES"
    fi
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Dual Encoder E/P/D (CPU+XPU encoders)" "$MODEL_NAME" "$HTTP_PORT"

# Start frontend
echo "Starting frontend..."
python -m dynamo.frontend &

EXTRA_ARGS=""
PD_EXTRA_ARGS=""

# GPU assignments
DYN_ENCODE_WORKER_1_GPU=${DYN_ENCODE_WORKER_1_GPU:-0}  # For XPU encoder
DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-1}
DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-2}

DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}

if [[ "${DEVICE_PLATFORM,,}" == "xpu" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --block-size 64"
    PD_EXTRA_ARGS="$PD_EXTRA_ARGS --max-model-len 10240"
fi

echo ""
echo "=========================================="
echo "Dual Encoder Configuration:"
echo "  - Encoder 1: CPU (vision model on CPU)"
echo "  - Encoder 2: XPU $DYN_ENCODE_WORKER_1_GPU (vision model on GPU)"
echo "  - Prefill: XPU $DYN_PREFILL_WORKER_GPU"
echo "  - Decode: XPU $DYN_DECODE_WORKER_GPU"
echo "  - Scheduler Mode: ${DYN_ENCODER_SCHEDULER:-per_request}"
echo "  - Split Ratio: ${DYN_ENCODER_SPLIT_RATIO:-1:1}"
echo "=========================================="
echo ""

# Start encoder worker 1 on CPU
echo "Starting encoder worker 1 with CPU vision model..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYN_ENCODER_DEVICE=cpu \
VLLM_ENCODER=0 \
env $DEVICE_AFFINITY_ENV=0 \
python -m dynamo.vllm \
  --multimodal-encode-worker \
  --enable-multimodal \
  --enable-mm-embeds \
  --model $MODEL_NAME \
  --gpu-memory-utilization 0.1 \
  $EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device": "cpu"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

# Start encoder worker 2 on XPU
echo "Starting encoder worker 2 with XPU vision model (GPU $DYN_ENCODE_WORKER_1_GPU)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20100 \
env $DEVICE_AFFINITY_ENV=$DYN_ENCODE_WORKER_1_GPU \
python -m dynamo.vllm \
  --multimodal-encode-worker \
  --enable-multimodal \
  --enable-mm-embeds \
  --model $MODEL_NAME \
  --gpu-memory-utilization 0.3 \
  $EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device": "'$DEVICE_PLATFORM'"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20083"}' &

sleep 5  # Give encoders time to register

# Start prefill worker (routes to both encoders via scheduler)
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
DYN_ENCODER_SCHEDULER="${DYN_ENCODER_SCHEDULER:-per_request}" \
DYN_ENCODER_SPLIT_RATIO="${DYN_ENCODER_SPLIT_RATIO:-1:1}" \
env $DEVICE_AFFINITY_ENV=$DYN_PREFILL_WORKER_GPU \
python -m dynamo.vllm \
  --multimodal-worker \
  --route-to-encoder \
  --disaggregation-mode prefill \
  --enable-multimodal \
  --enable-mm-embeds \
  --model $MODEL_NAME \
  --gpu-memory-utilization $DYN_PREFILL_GPU_MEM \
  $EXTRA_ARGS \
  $PD_EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device": "'$DEVICE_PLATFORM'"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

# Start decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
env $DEVICE_AFFINITY_ENV=$DYN_DECODE_WORKER_GPU \
python -m dynamo.vllm \
  --multimodal-decode-worker \
  --enable-multimodal \
  --enable-mm-embeds \
  --model $MODEL_NAME \
  --gpu-memory-utilization $DYN_DECODE_GPU_MEM \
  $EXTRA_ARGS \
  $PD_EXTRA_ARGS \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device": "'$DEVICE_PLATFORM'"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

echo ""
echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="
echo ""
echo "Scheduler: ${DYN_ENCODER_SCHEDULER:-per_request} mode (split: ${DYN_ENCODER_SPLIT_RATIO:-1:1})"
echo ""

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
