#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
dynamo run in=http out=dyn &

MODEL_NAME="llava-hf/llava-1.5-7b-hf"

# run processor
python3 components/processor.py --model $MODEL_NAME --prompt-template "USER: <image>\n<prompt> ASSISTANT:" &

# run E/P/D workers
CUDA_VISIBLE_DEVICES=0 python3 components/encode_worker.py --model $MODEL_NAME &
CUDA_VISIBLE_DEVICES=1 python3 components/worker.py --model $MODEL_NAME --worker-type prefill --enable-disagg &
CUDA_VISIBLE_DEVICES=2 python3 components/worker.py --model $MODEL_NAME --worker-type decode --enable-disagg