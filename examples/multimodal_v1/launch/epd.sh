#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
dynamo run in=http out=dyn &

# run processor
python3 components/processor.py --model llava-hf/llava-1.5-7b-hf --prompt-template "USER: <image>\n<prompt> ASSISTANT:"

# run E/P/D workers
python3 components/encode_worker.py --model llava-hf/llava-1.5-7b-hf

python3 components/worker.py --model llava-hf/llava-1.5-7b-hf