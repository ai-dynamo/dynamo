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
            echo "  - Processor component (uses pre-tokenized input with ModelInput.Tokens)"
            echo "  - vLLM-native encoder worker (producer using ECConnector)"
            echo "  - Multimodal worker (consumer using ECConnector, aggregated P+D)"
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

# Start frontend
echo "Starting frontend..."
python -m dynamo.frontend  &

# Start EC Processor (uses pre-tokenized input with placeholder tokens)
echo "Starting EC Processor..."
python -m dynamo.vllm \
    --ec-processor \
    --enable-multimodal \
    --model $MODEL_NAME &

# Start vLLM-native encoder worker (ECConnector producer)
echo "Starting vLLM-native encoder worker (ECConnector producer) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
    --vllm-native-encoder-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --ec-connector-backend $EC_CONNECTOR_BACKEND \
    --ec-storage-path $EC_STORAGE_PATH \
    --connector none \
    --enforce-eager \
    --max-num-batched-tokens 114688 \
    --no-enable-prefix-caching &

# Start aggregated multimodal worker (ECConnector consumer, P+D combined)
echo "Starting aggregated multimodal worker (ECConnector consumer) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm \
    --multimodal-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --ec-consumer-mode \
    --ec-connector-backend $EC_CONNECTOR_BACKEND \
    --ec-storage-path $EC_STORAGE_PATH \
    --enable-mm-embeds \
    --connector none \
    --enforce-eager &

# Wait for all background processes to complete
wait

