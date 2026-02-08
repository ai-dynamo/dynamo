#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
EC_STORAGE_PATH="/tmp/dynamo_ec_cache"
EC_CONNECTOR_BACKEND="ECExampleConnector"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --ec-storage-path)
            EC_STORAGE_PATH=$2
            shift 2
            ;;
        --ec-connector-backend)
            EC_CONNECTOR_BACKEND=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Aggregated multimodal serving with vLLM-native encoder (ECConnector mode)"
            echo ""
            echo "This script launches:"
            echo "  - Frontend server"
            echo "  - vLLM-native encoder worker (producer using ECConnector)"
            echo "  - Multimodal worker (routes to encoder + consumer using ECConnector, aggregated P+D)"
            echo ""
            echo "Options:"
            echo "  --model <model_name>              Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  --ec-storage-path <path>          Path for ECConnector storage (default: $EC_STORAGE_PATH)"
            echo "  --ec-connector-backend <backend>  ECConnector backend class (default: $EC_CONNECTOR_BACKEND)"
            echo "  -h, --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --ec-storage-path /shared/encoder-cache"
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

# Create storage directory if it doesn't exist
mkdir -p "$EC_STORAGE_PATH"

echo "=================================================="
echo "Aggregated Multimodal Serving (vLLM-Native Encoder with ECConnector)"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "ECConnector Backend: $EC_CONNECTOR_BACKEND"
echo "Storage Path: $EC_STORAGE_PATH"
echo "=================================================="

# GPU assignments (override via environment variables)
DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}

# GPU memory utilization for workers
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.75}
DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.85}

# Start frontend
echo "Starting frontend..."
python -m dynamo.frontend  &

# Start vLLM-native encoder worker (ECConnector producer)
echo "Starting vLLM-native encoder worker (ECConnector producer) on GPU $DYN_ENCODE_WORKER_GPU (mem: $DYN_ENCODE_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python -m dynamo.vllm \
    --vllm-native-encoder-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --ec-connector-backend $EC_CONNECTOR_BACKEND \
    --ec-storage-path $EC_STORAGE_PATH \
    --connector none \
    --enforce-eager \
    --gpu-memory-utilization $DYN_ENCODE_GPU_MEM \
    --max-num-batched-tokens 114688 \
    --no-enable-prefix-caching &

# Start aggregated multimodal worker (routes to encoder + ECConnector consumer, P+D combined)
# The worker handles encoder routing (frontend to encoder workers) and inference
echo "Starting aggregated multimodal worker (routes to encoder + ECConnector consumer) on GPU $DYN_PD_WORKER_GPU (mem: $DYN_PD_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU python -m dynamo.vllm \
    --dyn-route-to-encoder \
    --multimodal-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --dyn-ec-consumer-mode \
    --ec-connector-backend $EC_CONNECTOR_BACKEND \
    --ec-storage-path $EC_STORAGE_PATH \
    --enable-mm-embeds \
    --connector none \
    --enforce-eager \
    --gpu-memory-utilization $DYN_PD_GPU_MEM &

# Wait for all background processes to complete
wait

