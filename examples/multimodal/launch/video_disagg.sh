#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/LLaVA-NeXT-Video-7B-hf"
PROMPT_TEMPLATE="USER: <video>\n<prompt> ASSISTANT:"
NUM_FRAMES_TO_SAMPLE=8

# run ingress
python -m dynamo.frontend --http-port=8000 &


# run processor
python3 components/processor.py --model $MODEL_NAME --prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0 python3 components/video_encode_worker.py --model $MODEL_NAME --num-frames-to-sample $NUM_FRAMES_TO_SAMPLE &
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=1 python3 components/worker.py --model $MODEL_NAME --worker-type prefill --enable-disagg &
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 CUDA_VISIBLE_DEVICES=2 python3 components/worker.py --model $MODEL_NAME --worker-type decode --enable-disagg &

# Wait for all background processes to complete
wait
