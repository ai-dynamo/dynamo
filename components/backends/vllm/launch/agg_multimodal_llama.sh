#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

trap 'echo Cleaning up...; kill 0' EXIT

MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run processor
python -m dynamo.vllm --multimodal-processor --model $MODEL_NAME --mm-prompt-template "<|image|>\n<prompt>" &
# Llama 4 doesn't support image embedding input, so use encode+prefill worker
# that handles image encoding inline
python -m dynamo.vllm --multimodal-encode-prefill-worker --model $MODEL_NAME --tensor-parallel-size=8 --max-model-len=208960 --gpu-memory-utilization 0.80 &

# Wait for all background processes to complete
wait
