#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated sequence-classification / pooling model serving.
# One --classify-worker registers ModelType.Classify | ModelType.Pooling, so
# the frontend mounts both POST /classify and POST /pooling (matching native
# vLLM, which exposes both for every pooling-runner model).
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default classification model: a small 3-class NLI cross-encoder
# (contradiction / entailment / neutral). Larger alternative matching the
# same task: `tasksource/ModernBERT-large-nli`.
MODEL="cross-encoder/nli-MiniLM2-L6-H768"

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
            echo "  --model <name>  Specify classification model (default: $MODEL)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.vllm."
            echo "Note: --runner pooling is set here (required for pooling models)."
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
print_launch_banner --no-curl "Launching Classify Worker (1 GPU)" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/classify \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "input": "A man is playing a sport. Some men are playing a sport."
    }'
CURL

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# Classification inputs are short (a sentence or sentence pair), so the model's
# native context length is left as-is — unlike embedding models (Qwen3-Embedding
# is 32K, which pushes vLLM's KV-cache pre-check into the GiB range), the
# BERT/RoBERTa/ModernBERT classifiers here are 512-8192 positions and don't
# blow up the pre-check. Setting a fixed cap here would also break small
# models whose max_position_embeddings is below the cap (e.g. the default
# MiniLM classifier is 512). Override per-model via MAX_MODEL_LEN if needed.
MAX_MODEL_LEN_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

# run worker
# --classify-worker: registers ModelType.Classify | ModelType.Pooling.
# --runner pooling: required for pooling models.
# Prefix caching is left off by default for pooling-family workers (the worker
# defaults it to False; force-enabling it crashes hybrid-attention pooling
# models such as ModernBERT).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --classify-worker \
    --model "$MODEL" \
    --runner pooling \
    --trust-remote-code \
    "${MAX_MODEL_LEN_ARGS[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
