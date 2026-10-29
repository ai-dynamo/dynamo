#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated rerank serving through vLLM's scoring path.
# One --rerank-worker registers ModelType.Rerank, so the frontend mounts
# POST /v1/rerank.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"     # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Small MS MARCO cross-encoder with a single relevance logit.
MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h | --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>  Specify rerank model (default: $MODEL)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.vllm."
            echo "Note: --runner pooling is set here and required for rerank models."
            echo ""
            echo "Decoder-only rerankers need vLLM's score template and overrides, e.g.:"
            echo "  $0 --model Qwen/Qwen3-Reranker-4B \\"
            echo "    --custom-jinja-template /path/to/qwen3_reranker.jinja \\"
            echo "    --hf-overrides '{\"architectures\": [\"Qwen3ForSequenceClassification\"], \"classifier_from_token\": [\"no\", \"yes\"], \"is_original_qwen3_reranker\": true}'"
            echo "The template is vLLM's examples/pooling/score/template/qwen3_reranker.jinja."
            trap - EXIT
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
print_launch_banner --no-curl "Launching Rerank Worker (1 GPU)" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/rerank \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "query": "What is Dynamo?",
      "documents": ["Dynamo serves distributed inference.", "A recipe for bread."],
      "top_n": 1
    }'
CURL

# Run ingress.
python3 -m dynamo.frontend &

# Rerank models have varied native context lengths. Leave the model's own
# value unchanged unless the caller explicitly supplies a safe override.
MAX_MODEL_LEN_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

# Run a pooling AsyncLLM worker. Prefix caching remains disabled by the
# pooling-family worker default because pooling requests have no decode phase.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --rerank-worker \
    --model "$MODEL" \
    --runner pooling \
    "${MAX_MODEL_LEN_ARGS[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on the first process failure; the EXIT trap tears down the remainder.
wait_any_exit
