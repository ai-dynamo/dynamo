#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal (image) serving via the OpenEngine sidecar (1 GPU).
#
# Sidecar analogue of agg_multimodal.sh: instead of running vLLM in-process
# inside the Dynamo worker, this launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. A native `vllm-rs serve` engine (a VLM) exposing the OpenEngine gRPC
#      server
#   3. The Dynamo vLLM sidecar worker, which talks to (2) over OpenEngine
#
# The engine loads a vision-language model; the Rust frontend resolves its
# `MultimodalModelInfo` automatically (no flag needed) and reports
# supports_multimodal=true over `GetModelInfo`, which the sidecar discovers.
# The Dynamo frontend runs in URL-passthrough mode (media_decoder unset), so it
# forwards image URLs / data: URIs untouched; the engine fetches, decodes, and
# preprocesses them and expands the placeholder markers carried in the prompt
# token IDs. No NIXL/encoder transfer is involved for aggregated serving.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, print_curl_footer, wait_any_exit

# Default model: LLaVA-1.5, whose single `<image>` placeholder is a real
# tokenizer token (id 32000), so the marker-expansion path in llm-multimodal
# resolves. Qwen-VL does NOT work with this frontend: its llm-multimodal spec
# hardcodes the marker `<image>`, but the real Qwen tokenizer only has
# `<|image_pad|>`, so backend init fails resolving the marker token id.
MODEL="${DYN_MODEL_NAME:-llava-hf/llava-1.5-7b-hf}"

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
            echo "  --model <name>       Specify the VLM model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional options are passed through to the vllm-rs serve engine."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
OPENENGINE_HOST="${OPENENGINE_HOST:-127.0.0.1}"
OPENENGINE_PORT="${OPENENGINE_PORT:-50051}"
# vllm-rs runs its own OpenAI HTTP frontend; it is unused by the sidecar but
# still binds, so keep it off the Dynamo frontend's port (8000).
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Use TCP transport (instead of default NATS): multimodal base64 images in
# data: URIs can exceed the NATS 1MB max payload limit.
export DYN_REQUEST_PLANE=tcp

print_launch_banner --no-curl "Launching Sidecar Aggregated Multimodal Serving (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Backend:     dynamo-vllm-sidecar over OpenEngine (URL-passthrough)" \
    "Media:       image_url (http(s) URL or data: URI)"

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

# 2. Native vLLM engine (Rust frontend + managed Python EngineCore) with the
# OpenEngine gRPC server, loading the VLM. The model is positional;
# --enforce-eager is for quick startup (drop it for production).
vllm-rs serve "$MODEL" \
    --port "$VLLM_RS_HTTP_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --engine-rpc-host "$OPENENGINE_HOST" \
    --engine-rpc-port "$OPENENGINE_PORT" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no vllm import; OpenEngine client only). It receives
# ONLY the endpoint; model, role, and multimodal support are discovered from the
# engine.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
