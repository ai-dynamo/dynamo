#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Demo: vLLM serve with Dynamo aggregated embedding cache (vLLM 0.17+)
#
# This script launches a vLLM server with Dynamo's CPU-side LRU embedding
# cache enabled, then sends sample requests to demonstrate cache hits.
#
# Usage:
#   ./demo_vllm_embedding_cache.sh [--model MODEL] [--capacity-gb GB] [--port PORT]
#
# Requirements:
#   - vLLM >= 0.18.0 (ec_both ECConnector role is native)
#   - dynamo installed (provides DynamoMultimodalEmbeddingCacheConnector)
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
CAPACITY_GB="${CAPACITY_GB:-10}"
PORT="${PORT:-8000}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --capacity-gb)  CAPACITY_GB="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

echo "============================================"
echo " vLLM + Dynamo Embedding Cache Demo"
echo "============================================"
echo " Model:        ${MODEL}"
echo " Cache:        ${CAPACITY_GB} GB"
echo " Port:         ${PORT}"
echo "============================================"
echo ""

# --- Step 1: Launch vLLM server ---
echo "==> Starting vLLM server with embedding cache..."

EC_CONFIG="{
    \"ec_role\": \"ec_both\",
    \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
    \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
    \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": ${CAPACITY_GB}}
}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --ec-transfer-config "$EC_CONFIG" \
    "${EXTRA_ARGS[@]}" &

SERVER_PID=$!
trap "echo '==> Stopping server...'; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

# --- Step 2: Wait for server readiness ---
echo "==> Waiting for server to be ready on port ${PORT}..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "==> Server ready."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process exited unexpectedly."
        exit 1
    fi
    sleep 2
done

if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: Server did not become ready within 240 seconds."
    exit 1
fi

# --- Step 3: Send requests (same image = cache hit on 2nd+) ---
echo ""
echo "==> Sending request 1 (cache miss — first time seeing image)..."
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"max_tokens\": 64,
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
                {\"type\": \"text\", \"text\": \"What is in this image?\"}
            ]
        }]
    }" | python3 -m json.tool

echo ""
echo "==> Sending request 2 (cache hit — same image)..."
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"max_tokens\": 64,
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
                {\"type\": \"text\", \"text\": \"Describe the colors in this image.\"}
            ]
        }]
    }" | python3 -m json.tool

echo ""
echo "==> Sending request 3 (cache hit — same image, different question)..."
curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"max_tokens\": 64,
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
                {\"type\": \"text\", \"text\": \"Is there any transparency in this image?\"}
            ]
        }]
    }" | python3 -m json.tool

echo ""
echo "============================================"
echo " Demo complete."
echo " Request 1: cache MISS (encoder runs)"
echo " Request 2-3: cache HIT (encoder skipped)"
echo " Check server logs for cache hit/miss stats."
echo "============================================"
