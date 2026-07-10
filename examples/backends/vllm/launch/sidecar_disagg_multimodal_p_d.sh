#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated 1P1D multimodal (image) serving via the vLLM gRPC sidecar
# (2 GPUs).
#
# Sidecar analogue of disagg_multimodal_p_d.sh. Five processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. Native vLLM decode engine  (kv_role=kv_consumer) + vLLM gRPC server
#   3. Dynamo decode sidecar worker -> talks to (2)
#   4. Native vLLM prefill engine (kv_role=kv_producer) + vLLM gRPC server
#   5. Dynamo prefill sidecar worker -> talks to (4)
#
# Both engines load the same VLM weights. The vision encoder runs *inside* the
# prefill engine: the media only needs to reach prefill, which encodes +
# prefills and produces the KV the decode peer pulls. KV moves over NIXL between
# the two engines exactly as in the text disagg path (no new encoder/EPD
# transfer); the sidecars only relay kv_transfer_params across
# the Dynamo boundary. The Dynamo prefill router sends the full request (with
# media) to prefill and a token-only request to decode.

set -e

# Common configuration: LLaVA-1.5, whose single `<image>` placeholder is a real
# tokenizer token (id 32000) that the llm-multimodal marker-expansion path can
# resolve. Qwen-VL does NOT work with this frontend (its spec hardcodes the
# marker `<image>`, absent from the real Qwen tokenizer, which only has
# `<|image_pad|>`), so backend init fails.
MODEL="${DYN_MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo "  Launches a 1P1D multimodal sidecar deployment (2 GPUs)."
    echo "  Override the model with DYN_MODEL_NAME=<vlm>."
    exit 0
fi

trap 'echo Cleaning up...; kill 0' EXIT

# vLLM gRPC ports (one per engine).
DECODE_GRPC_PORT="${DECODE_GRPC_PORT:-50051}"
PREFILL_GRPC_PORT="${PREFILL_GRPC_PORT:-50052}"
# vllm-rs runs its own OpenAI HTTP frontend per engine; unused by the sidecar
# but still binds, so keep each off the Dynamo frontend's port (8000).
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-8100}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-8101}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Use TCP transport (instead of default NATS): multimodal base64 images in
# data: URIs can exceed the NATS 1MB max payload limit.
export DYN_REQUEST_PLANE=tcp

print_launch_banner --no-curl "Launching Sidecar Disaggregated Multimodal Serving (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Backend:     dynamo-vllm-sidecar 1P1D over vLLM gRPC (URL-passthrough)" \
    "Media:       image_url (http(s) URL or data: URI); encode runs in prefill"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}
      ]}],
      "max_tokens": 50
    }'
CURL

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Decode engine: vllm-rs serve, kv_consumer, vLLM gRPC on DECODE_GRPC_PORT.
# --kv-transfer-config is forwarded to the Python EngineCore.
# --enforce-eager is for quick startup; drop it for production.
CUDA_VISIBLE_DEVICES=0 vllm-rs serve "$MODEL" \
    --port "$DECODE_HTTP_PORT" \
    --grpc-port "$DECODE_GRPC_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &

# 3. Decode sidecar worker (endpoint-only; role discovered as DECODE).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${DECODE_GRPC_PORT}" &

# 4. Prefill engine: vllm-rs serve, kv_producer, vLLM gRPC on PREFILL_GRPC_PORT.
# The vision encoder runs here. KV events published so KV-aware routing can
# observe the prefill cache.
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 vllm-rs serve "$MODEL" \
    --port "$PREFILL_HTTP_PORT" \
    --grpc-port "$PREFILL_GRPC_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# 5. Prefill sidecar worker (endpoint-only; role discovered as PREFILL).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${PREFILL_GRPC_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
