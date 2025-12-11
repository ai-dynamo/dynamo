#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64

# run frontend + KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# run workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8082 \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --connector none \
    --enable-lora \
    --max-lora-rank 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --connector none \
    --enable-lora \
    --max-lora-rank 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'

# below commands are not executed automatically in the script because previous backend launch command is blocking.

################################## Example Usage ##################################

# Check available models
curl http://localhost:8000/v1/models | jq .

# Load LoRA to instances using s3 uri
curl -s  -X POST http://localhost:8081/v1/loras \
       -H "Content-Type: application/json" \
       -d '{"lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
     "source": {"uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"}}' | jq .

curl -s  -X POST http://localhost:8082/v1/loras \
       -H "Content-Type: application/json" \
       -d '{"lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
     "source": {"uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"}}' | jq .

 # Test LoRA inference
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }' | jq .


 # Sample output after running above curl request twice.
 # usage.prompt_tokens_details.cached_tokens is the number of tokens that were cached from the previous request.
{
  "id": "chatcmpl-0cf880c2-fe98-45c4-9c76-84c3ad1a56cc",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "<think>\nOkay, so I need to develop a character background for a character named Elara. Let me start by understanding the requirements. The user wants",
        "role": "assistant",
        "reasoning_content": null
      },
      "finish_reason": "length"
    }
  ],
  "created": 1765230243,
  "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 196,
    "completion_tokens": 30,
    "total_tokens": 226,
    "prompt_tokens_details": {
      "audio_tokens": null,
      "cached_tokens": 192              # tokens that were cached from the previous request.
    }
  },
  "nvext": {
    "worker_id": {
      "prefill_worker_id": 7587891281668871552,
      "decode_worker_id": 7587891281668871552
    }
  }
}