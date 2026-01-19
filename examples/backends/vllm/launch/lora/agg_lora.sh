#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ALLOW_HTTP=true

# Dynamo LoRA Configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio

mkdir -p $DYN_LORA_PATH

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var.
python -m dynamo.frontend &

# run worker
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager  \
    --connector none  \
    --enable-lora  \
    --max-lora-rank 64

################################## Example Usage ##################################

# Check available models
curl http://localhost:8000/v1/models | jq .

# Load LoRA using s3 uri
curl -s  -X POST http://localhost:8081/v1/loras \
       -H "Content-Type: application/json" \
       -d '{"lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
     "source": {"uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"}}' | jq .

# Test LoRA inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 300,
    "temperature": 0.0
  }'

# Test base model inference (for comparison)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 300,
    "temperature": 0.0
  }'

# Unload LoRA
curl -X DELETE http://localhost:8081/v1/loras/codelion/Qwen3-0.6B-accuracy-recovery-lora
