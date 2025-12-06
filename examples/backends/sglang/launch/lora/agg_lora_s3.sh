#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Follow the README.md instructions to setup MinIO or upload the LoRA to s3/minio
# Adjust these values to match your local MinIO or S3 setup


# load math lora to minio
# LORA_NAME=codelion/Qwen3-0.6B-accuracy-recovery-lora HF_LORA_REPO=codelion/Qwen3-0.6B-accuracy-recovery-lora ./setup_minio.sh


export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ALLOW_HTTP=true

# Dynamo LoRA Configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio
export DYN_LOG=debug
# export DYN_LOG_LEVEL=debug

mkdir -p $DYN_LORA_PATH
MODEL="Qwen/Qwen3-0.6B"
# run ingress
python -m dynamo.frontend --http-port=8000 &


DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-lora  \
  --max-lora-rank 32 \
  --lora-target-modules all

################################## Example Usage ##################################

# Check available models
curl http://localhost:8000/v1/models | jq .

# Load LoRA using s3 uri
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "YorkFr/financial-sentiment-qwen3-v2",
    "source": {
      "uri": "s3://my-loras/YorkFr/financial-sentiment-qwen3-v2"
    }
  }'

# Test LoRA inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Neural-Hacker/Qwen3-Math-Reasoning-LoRA",
    "messages": [{"role": "user", "content": "Solve (x*x - x + 1 = 0) for x"}],
    "max_tokens": 300,
    "temperature": 0.0
  }'

# Find the minimum possible value of \( x^2 + y^2 \) given that \( x \) and \( y \) are real numbers satisfying \( xy(x^2 - y^2) = x^2 + y^2 \) and \( x \neq 0 \)
# Test base model inference (for comparison)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Solve (x*x - x + 1 = 0) for x"}],
    "max_tokens": 300,
    "temperature": 0.0
  }'

# Unload LoRA
curl -X DELETE http://localhost:8081/v1/loras/Neural-Hacker/Qwen3-Math-Reasoning-LoRA
