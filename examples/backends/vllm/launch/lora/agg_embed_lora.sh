#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated embedding serving with LoRA adapters.
# Adapters apply only to architectures vLLM declares SupportsLoRA. Qwen3-Embedding
# is decoder-backed and qualifies; encoder-backed embedders (BGE, E5, BERT,
# RoBERTa) do not, and vLLM refuses to start with --enable-lora on those.
# GPUs: 1
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ALLOW_HTTP=true

# Dynamo LoRA Configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio_embed

mkdir -p $DYN_LORA_PATH

MODEL="Qwen/Qwen3-Embedding-0.6B"
SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching Aggregated Embeddings + LoRA (1 GPU)" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  # Load an adapter
  curl -s -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\
    -H 'Content-Type: application/json' \\
    -d '{"lora_name": "my-adapter", "source": {"uri": "s3://my-loras/my-adapter"}}'

  # Embed through the adapter
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "my-adapter", "input": "The capital of France is Paris."}'

  # Embed with the base model, for comparison
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "${MODEL}", "input": "The capital of France is Paris."}'
CURL

# run ingress
python -m dynamo.frontend &

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
# --runner pooling: required for embedding models.
# --max-lora-rank 64 matches the reference adapter; raise it for wider adapters.
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT} \
    python -m dynamo.vllm --model "$MODEL" \
    --embedding-worker \
    --runner pooling \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    --enable-lora \
    --max-lora-rank 64 &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
