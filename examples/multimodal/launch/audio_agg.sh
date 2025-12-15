#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            echo "  --prompt-template <template> Specify the multi-modal prompt template to use. LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates."
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
    echo "Example: --prompt-template 'USER: <image>\n<prompt> ASSISTANT:'"
    exit 1
fi

# Check and install required dependencies for audio multimodal models
echo "Checking audio multimodal dependencies..."
DEPS_MISSING=false

# Check for accelerate
if ! python -c "import accelerate" &> /dev/null; then
    echo "  accelerate not found"
    DEPS_MISSING=true
else
    echo "  ✓ accelerate is installed"
fi

# Check for vllm with audio support
if ! python -c "import vllm" &> /dev/null; then
    echo "  vllm not found"
    DEPS_MISSING=true
else
    # Check if audio dependencies are available (librosa is a key audio dependency)
    if ! python -c "import librosa" &> /dev/null; then
        echo "  vllm audio dependencies not found"
        DEPS_MISSING=true
    else
        echo "  ✓ vllm with audio support is installed"
    fi
fi

# Install missing dependencies
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
python3 components/processor.py --model $MODEL_NAME --prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers
CUDA_VISIBLE_DEVICES=0 python3 components/audio_encode_worker.py --model $MODEL_NAME &
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 python3 components/worker.py --model $MODEL_NAME --worker-type prefill &

# Wait for all background processes to complete
wait
