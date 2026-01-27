#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/LLaVA-NeXT-Video-7B-hf"
PROMPT_TEMPLATE="USER: <video>\n<prompt> ASSISTANT:"
NUM_FRAMES_TO_SAMPLE=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --num-frames-to-sample)
            NUM_FRAMES_TO_SAMPLE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name>           Specify the model to use (default: $MODEL_NAME)"
            echo "  --num-frames-to-sample <count> Number of frames to sample (default: $NUM_FRAMES_TO_SAMPLE)"
            echo "  -h, --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run processor
python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME --mm-prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-video-encode-worker --enable-multimodal --model $MODEL_NAME --num-frames-to-sample $NUM_FRAMES_TO_SAMPLE --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --multimodal-worker --is-prefill-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 CUDA_VISIBLE_DEVICES=2 python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

# Wait for all background processes to complete
wait
