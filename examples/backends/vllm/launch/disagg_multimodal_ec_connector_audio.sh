#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
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
            echo "Disaggregated multimodal audio serving with vLLM-native encoder (ECConnector mode)"
            echo ""
            echo "This script launches:"
            echo "  - Frontend server"
            echo "  - EC Processor (pre-tokenized input with ModelInput.Tokens)"
            echo "  - vLLM-native encoder worker (ECConnector producer)"
            echo "  - Prefill worker (ECConnector consumer)"
            echo "  - Decode worker"
            echo ""
            echo "Options:"
            echo "  --model <model_name>              Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  --ec-storage-path <path>          Path for ECConnector storage (default: $EC_STORAGE_PATH)"
            echo "  --ec-connector-backend <backend>  ECConnector backend class (default: $EC_CONNECTOR_BACKEND)"
            echo "  -h, --help                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check and install required dependencies for audio multimodal models
echo "Checking audio multimodal dependencies..."
DEPS_MISSING=false

if ! python -c "import accelerate" &> /dev/null; then
    echo "  accelerate not found"
    DEPS_MISSING=true
else
    echo "  ✓ accelerate is installed"
fi

if ! python -c "import vllm" &> /dev/null; then
    echo "  vllm not found"
    DEPS_MISSING=true
else
    if ! python -c "import librosa" &> /dev/null; then
        echo "  vllm audio dependencies not found"
        DEPS_MISSING=true
    else
        echo "  ✓ vllm with audio support is installed"
    fi
fi

if [ "$DEPS_MISSING" = true ]; then
    echo "Installing missing dependencies..."
    pip install 'vllm[audio]' accelerate
    echo "Dependencies installed successfully"
else
    echo "All required dependencies are already installed"
fi

mkdir -p "$EC_STORAGE_PATH"

echo "=================================================="
echo "Disaggregated Multimodal Audio Serving (ECConnector)"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "ECConnector Backend: $EC_CONNECTOR_BACKEND"
echo "Storage Path: $EC_STORAGE_PATH"
echo "=================================================="

python -m dynamo.frontend &

python -m dynamo.vllm \
    --ec-processor \
    --enable-multimodal \
    --model $MODEL_NAME &

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

VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm \
    --multimodal-worker \
    --is-prefill-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --ec-consumer-mode \
    --ec-connector-backend $EC_CONNECTOR_BACKEND \
    --ec-storage-path $EC_STORAGE_PATH \
    --enable-mm-embeds \
    --enforce-eager \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=2 python -m dynamo.vllm \
    --multimodal-decode-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --enable-mm-embeds \
    --enforce-eager \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

wait
