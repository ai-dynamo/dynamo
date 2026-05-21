#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated embedding model serving.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default embedding model. Smaller alternatives:
# `BAAI/bge-small-en-v1.5` (CPU-friendly), `intfloat/e5-small-v2`.
MODEL="Qwen/Qwen3-Embedding-0.6B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>  Specify embedding model (default: $MODEL)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.vllm."
            echo "Note: --runner pooling, --dtype float32, and --pooler-config"
            echo "are set here. Override via EXTRA_ARGS if your model requires"
            echo "different pooling."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching Embedding Worker (1 GPU)" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "input": "Hello, world!"
    }'
CURL

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# Tunable: most embedding workloads have short inputs (typically
# 60-200 tokens; OpenAI's text-embedding-3 cap is 8K). Defaulting to
# the model's native max (32K for Qwen3-Embedding) would push vLLM's
# KV-cache pre-check into the GiB range even though pooling has no
# real KV cache. Override via MAX_MODEL_LEN env var if you have
# unusually long embedding inputs.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

# run worker
# --runner pooling: required for embedding models.
# --pooler-config: MEAN pool, no activation — the Qwen3-Embedding default.
# --dtype float32: matches Qwen3-Embedding's released weights; override
#   if your model is shipped at a different precision.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --embedding-worker \
    --model "$MODEL" \
    --runner pooling \
    --dtype float32 \
    --pooler-config '{"pooling_type": "MEAN", "use_activation": false}' \
    --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching \
    --trust-remote-code \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
