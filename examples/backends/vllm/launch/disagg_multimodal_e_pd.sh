#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E/PD (Encode + aggregated Prefill-Decode) multimodal deployment
#
# Architecture: 2-component disaggregation (no separate processor process)
# - Encode Worker: Dedicated vision encoder that extracts image embeddings
# - PD Worker: Aggregated prefill/decode worker that receives requests from
#              the frontend, routes images to encode workers, and runs full
#              inference. (Uses --dyn-route-to-encoder for encode routing)
#
# Benefits: Decouples vision encoding from LLM inference, enables independent
#           scaling of encode vs inference, one fewer process than the old
#           Processor+Encode+PD topology.
# For full P/D disaggregation, see disagg_multimodal_epd.sh
# For standard single-worker deployment, see agg_multimodal.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
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
            echo "E/PD multimodal serving: separate Encode worker + aggregated PD worker"
            echo ""
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --single-gpu         Run both workers on GPU 0 (for pre-merge CI)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --single-gpu"
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
echo "E/PD Multimodal Serving"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "=================================================="

# Start frontend (HTTP endpoint)
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Set max model length based on model name (respect env var override)
if [[ -z "$DYN_MAX_MODEL_LEN" ]]; then
    if [[ "$MODEL_NAME" == "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" ]]; then
        DYN_MAX_MODEL_LEN="16384"
    elif [[ "$MODEL_NAME" == "llava-hf/llava-1.5-7b-hf" ]]; then
        DYN_MAX_MODEL_LEN="2048"
    else
        DYN_MAX_MODEL_LEN="30426"
    fi
fi

# Set GPU memory utilization based on deployment mode
# Single-GPU mode: Both workers share GPU 0, so use reduced memory settings
# Multi-GPU mode: Each worker gets its own GPU, so use higher memory settings
EXTRA_ARGS=""
if [[ "$SINGLE_GPU" == "true" ]]; then
    EXTRA_ARGS="--gpu-memory-utilization 0.4 --enforce-eager --max-model-len $DYN_MAX_MODEL_LEN"
else
    EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len $DYN_MAX_MODEL_LEN"
fi

# Launch Encode Worker + PD Worker
if [[ "$SINGLE_GPU" == "true" ]]; then
    # Single GPU mode: both workers share GPU 0 with reduced memory
    echo "Starting encode worker on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME $EXTRA_ARGS &
    # Stagger startup to avoid concurrent GPU allocation on the same device
    sleep 60
    echo "Starting PD worker on GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --dyn-route-to-encoder --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS &
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.75}
    echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (mem: $DYN_ENCODE_GPU_MEM)..."
    CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME --gpu-memory-utilization $DYN_ENCODE_GPU_MEM --max-model-len $DYN_MAX_MODEL_LEN &

    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}
    echo "Starting PD worker on GPU $DYN_PD_WORKER_GPU ..."
    CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU python -m dynamo.vllm --dyn-route-to-encoder --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS &
fi

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait
