#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
# Chose Qwen3-30B because its a small MOE that can fit on smaller GPUs (L40S for example)
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m dynamo.vllm \
    --model Qwen/Qwen3-30B-A3B \
    --data-parallel-rank $i \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --enforce-eager &
done

echo "All workers starting. (press Ctrl+C to stop)..."
wait
