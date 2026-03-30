#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

export VLLM_TARGET_DEVICE=xpu

# Default values
MODEL_NAME="llava-hf/LLaVA-NeXT-Video-7B-hf"
PROMPT_TEMPLATE="USER: <video>\n<prompt> ASSISTANT:"
NUM_FRAMES_TO_SAMPLE=8
MAX_MODEL_LEN=8192 # reduce KV cache memory usage to fit within B60 VRAM

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run processor
python3 $SCRIPT_DIR/../../components/processor.py --model $MODEL_NAME --prompt-template "$PROMPT_TEMPLATE" &

# run E/P/D workers
GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL_NAME")

ZE_AFFINITY_MASK=0 python3 $SCRIPT_DIR/../../components/video_encode_worker.py \
--model $MODEL_NAME \
--num-frames-to-sample $NUM_FRAMES_TO_SAMPLE &

VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=1 python3 $SCRIPT_DIR/../../components/worker.py --model $MODEL_NAME \
--worker-type prefill \
--max-model-len $MAX_MODEL_LEN \
${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} &

# Wait for all background processes to complete
wait
