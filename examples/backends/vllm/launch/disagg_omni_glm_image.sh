#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 2-stage disaggregated GLM-Image text-to-image generation.
# Stage 0: AR  (GPU 0) — generates prior_token_ids
# Stage 1: DiT (GPU 1) — diffusion denoising + VAE decode → image
# Router: orchestrates the 2-stage pipeline, formats response
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-zai-org/GLM-Image}"

# Resolve vllm-omni's built-in GLM-Image stage config
if [ -z "$STAGE_CONFIG" ]; then
    STAGE_CONFIG="$(python -c "import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'model_executor/stage_configs/glm_image.yaml'))" 2>/dev/null | tail -1)"
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --stage-configs-path) STAGE_CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
# Use an isolated namespace by default to avoid stale discovery/model-card
# collisions from previous disaggregated runs (which can route directly to dit).
if [ -z "${DYN_NAMESPACE:-}" ]; then
    export DYN_NAMESPACE="dynamo-omni-glm-$(date +%s)"
fi
echo "Namespace:   ${DYN_NAMESPACE}"
print_launch_banner --no-curl "Disaggregated GLM-Image (2-stage, 2 GPUs)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/images/generations \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "a red apple on a white table",
    "size": "1024x1024"
  }' | jq
CURL

export FLASHINFER_DISABLE_VERSION_CHECK=1

# Stage 0: AR worker (GPU 0) — generates prior_token_ids
echo "Starting Stage 0 (AR)..."
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 0 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities image \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Stage 1: DiT worker (GPU 1) — diffusion denoising + VAE decode
echo "Starting Stage 1 (DiT)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8082 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 1 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities image \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Router — discovers stage workers, orchestrates pipeline, formats response
echo "Starting Router..."
DYN_SYSTEM_PORT=8083 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --omni-router \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities image \
    --media-output-fs-url file:///tmp/dynamo_media \
    "${EXTRA_ARGS[@]}" &
sleep 5

# Frontend
echo "Starting Frontend..."
python -m dynamo.frontend &

wait_any_exit
