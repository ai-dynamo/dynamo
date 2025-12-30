#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
PROMPT_TEMPLATE="USER: <image>\n<prompt> ASSISTANT:"
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
            echo ""
            echo "Disaggregated multimodal serving with separate Encode/Prefill/Decode workers"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  --prompt-template <template>  Specify the multi-modal prompt template to use"
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

# Set PROMPT_TEMPLATE based on the MODEL_NAME
if [[ -n "$PROVIDED_PROMPT_TEMPLATE" ]]; then
    PROMPT_TEMPLATE="$PROVIDED_PROMPT_TEMPLATE"
elif [[ "$MODEL_NAME" == "llava-hf/llava-1.5-7b-hf" ]]; then
    PROMPT_TEMPLATE="USER: <image>\n<prompt> ASSISTANT:"
elif [[ "$MODEL_NAME" == "microsoft/Phi-3.5-vision-instruct" ]]; then
    PROMPT_TEMPLATE="<|user|>\n<|image_1|>\n<prompt><|end|>\n<|assistant|>\n"
elif [[ "$MODEL_NAME" == "Qwen/Qwen2.5-VL-7B-Instruct" ]]; then
    PROMPT_TEMPLATE="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><prompt><|im_end|>\n<|im_start|>assistant\n"
else
    echo "No multi-modal prompt template is defined for the model: $MODEL_NAME"
    echo "Please provide a prompt template using --prompt-template option."
    echo "Example: --prompt-template 'USER: <image>\n<prompt> ASSISTANT:'"
    exit 1
fi

echo "=================================================="
echo "Disaggregated Multimodal Serving"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Prompt Template: $PROMPT_TEMPLATE"
echo "=================================================="


# Start frontend (no router mode)
echo "Starting frontend..."
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Start processor
echo "Starting processor..."
python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME --mm-prompt-template "$PROMPT_TEMPLATE" &

EXTRA_ARGS=""

# Start encode worker
echo "Starting encode worker on GPU 0..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME  $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

# Start prefill worker
echo "Starting prefill worker on GPU 1..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --multimodal-worker --is-prefill-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

# Start decode worker
echo "Starting decode worker on GPU 2..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=2 python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --model $MODEL_NAME $EXTRA_ARGS --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait

