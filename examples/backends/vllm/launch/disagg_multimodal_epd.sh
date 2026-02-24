#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

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
            echo "Disaggregated multimodal serving with separate Encode/Prefill/Decode workers"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
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


echo "=================================================="
echo "Disaggregated Multimodal Serving"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "=================================================="


# Start frontend (no router mode)
echo "Starting frontend..."
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

EXTRA_ARGS=""

# GPU assignments (override via environment variables)
DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-1}
DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-2}

# GPU memory utilization for workers
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}

# Start encode worker
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME --gpu-memory-utilization $DYN_ENCODE_GPU_MEM $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

# Start prefill worker (also handles encode routing via --route-to-encoder)
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU (GPU mem: $DYN_PREFILL_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU python -m dynamo.vllm --route-to-encoder --disaggregation-mode prefill --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --gpu-memory-utilization $DYN_PREFILL_GPU_MEM $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

# Start decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU (GPU mem: $DYN_DECODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --gpu-memory-utilization $DYN_DECODE_GPU_MEM $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait

