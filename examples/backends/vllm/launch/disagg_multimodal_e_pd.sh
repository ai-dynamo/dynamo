#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
SINGLE_GPU=false

# Parse command line arguments
# All extra arguments are passed through to the PD worker's dynamo.vllm
# (which routes them to Dynamo or vLLM as appropriate).
EXTRA_PD_ARGS=()
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
            echo "Usage: $0 [OPTIONS] [EXTRA_ARGS...]"
            echo ""
            echo "Disaggregated multimodal serving with separate Encode and aggregated PD worker"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  --single-gpu                  Run encode and PD workers on the same GPU (for small models, e.g. 2B)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "All additional arguments are passed through to the PD worker's dynamo.vllm."
            echo "Dynamo args (e.g. --multimodal-embedding-cache-capacity-gb) and"
            echo "vLLM engine args (e.g. --no-enable-prefix-caching) are automatically routed."
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
            echo "  $0 --no-enable-prefix-caching --multimodal-embedding-cache-capacity-gb 2"
            echo "  $0 --model Qwen/Qwen2-VL-2B-Instruct --single-gpu"
            echo ""
            exit 0
            ;;
        *)
            EXTRA_PD_ARGS+=("$1")
            shift
            ;;
    esac
done


PD_MAX_MODEL_LEN="16384"


echo "=================================================="
echo "Disaggregated Multimodal Serving (E + PD)"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "=================================================="


# Start frontend (no router mode)
echo "Starting frontend..."
python -m dynamo.frontend &

EXTRA_ARGS=""

# Embedding transfer: 1 = local file (safetensors), 0 = NIXL RDMA
export TRANSFER_LOCAL=${TRANSFER_LOCAL:-1}

# GPU assignments (override via environment variables)
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-0}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.4}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.4}
    EXTRA_ARGS="--enforce-eager"
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.9}
fi

# Start encode worker
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
python -m dynamo.vllm \
  --multimodal-encode-worker \
  --enable-multimodal \
  --model "$MODEL_NAME" \
  --gpu-memory-utilization "$DYN_ENCODE_GPU_MEM" \
  $EXTRA_ARGS &

# Start PD worker (aggregated prefill+decode, routes to encoder for embeddings)
echo "Starting PD worker on GPU $DYN_PD_WORKER_GPU (GPU mem: $DYN_PD_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU \
python -m dynamo.vllm \
  --route-to-encoder \
  --multimodal-worker \
  --enable-multimodal \
  --enable-mm-embeds \
  --model "$MODEL_NAME" \
  --max-model-len "$PD_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$DYN_PD_GPU_MEM" \
  $EXTRA_ARGS \
  "${EXTRA_PD_ARGS[@]}" &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait
