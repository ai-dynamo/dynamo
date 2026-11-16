#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated sequence-classification and pooling model serving.
# One --classify-worker registers both pooling-family model types, so the
# frontend mounts POST /v1/classify and POST /v1/pooling.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"     # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Small three-class NLI cross-encoder (contradiction / entailment / neutral).
MODEL="cross-encoder/nli-MiniLM2-L6-H768"

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
            echo "  --model <name>  Specify classification model (default: $MODEL)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.vllm."
            echo "Note: --runner pooling is set here and required for pooling models."
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
print_launch_banner --no-curl "Launching Classify Worker (1 GPU)" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/classify \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "input": "A man is playing a sport. Some men are playing a sport."
    }'
CURL

# Run ingress.
python3 -m dynamo.frontend &

# Classification models have varied native context lengths. Leave the model's
# own value unchanged unless the caller explicitly supplies a safe override.
MAX_MODEL_LEN_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

# Run a pooling AsyncLLM worker. Prefix caching remains disabled by the
# pooling-family worker default because pooling requests have no decode phase.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --classify-worker \
    --model "$MODEL" \
    --runner pooling \
    --trust-remote-code \
    "${MAX_MODEL_LEN_ARGS[@]}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on the first process failure; the EXIT trap tears down the remainder.
wait_any_exit
