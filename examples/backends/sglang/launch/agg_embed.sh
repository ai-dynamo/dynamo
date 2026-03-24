#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated embedding model serving.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL="Qwen/Qwen3-Embedding-4B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: System metrics are enabled by default on port 8081 (worker)"
            echo "Note: OpenTelemetry tracing is not yet supported for embedding models"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPU_MEM_FRACTION=$(build_gpu_mem_args sglang --model "$MODEL" 2>/dev/null || true)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching Embedding Worker" "$MODEL" "$HTTP_PORT"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/embeddings \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "input": "${EXAMPLE_PROMPT}"
    }'
CURL

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run worker
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --embedding-worker \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --use-sglang-tokenizer \
  ${GPU_MEM_FRACTION:+--mem-fraction-static "$GPU_MEM_FRACTION"} \
  --enable-metrics \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
