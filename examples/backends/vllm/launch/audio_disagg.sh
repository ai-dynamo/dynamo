#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
PROMPT_TEMPLATE=""
PROVIDED_PROMPT_TEMPLATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --prompt-template)
            PROVIDED_PROMPT_TEMPLATE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --prompt-template <template> Specify the multi-modal prompt template to use."
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set PROMPT_TEMPLATE based on the MODEL_NAME
if [[ -n "$PROVIDED_PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE="$PROVIDED_PROMPT_TEMPLATE"
elif [[ "$MODEL_NAME" == "Qwen/Qwen2-Audio-7B-Instruct" ]]; then
    PROMPT_TEMPLATE="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n<prompt><|im_end|>\n<|im_start|>assistant\n"
else
    echo "No multi-modal prompt template is defined for the model: $MODEL_NAME"
    echo "Please provide a prompt template using --prompt-template option."
    exit 1
fi

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

# run ingress
python -m dynamo.frontend --http-port 8000 &

# run processor
python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME --mm-prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-audio-encode-worker --enable-multimodal --model $MODEL_NAME --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --multimodal-worker --is-prefill-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 CUDA_VISIBLE_DEVICES=2 python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

# Wait for all background processes to complete
wait
