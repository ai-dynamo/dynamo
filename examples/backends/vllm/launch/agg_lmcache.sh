#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let LMCache or Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

MODEL="Qwen/Qwen3-0.6B"
# --block-size 64 is required for XPU; on CUDA vLLM uses its default
if [[ "${DYN_DEVICE:-cuda}" == "xpu" ]]; then
    BLOCK_SIZE_ARG=(--block-size "${DYN_BLOCK_SIZE:-64}")
else
    BLOCK_SIZE_ARG=()
fi
# KV buffer device: set DYN_DEVICE=xpu for Intel XPU hardware (default: cuda)
KV_BUFFER_DEVICE="${DYN_DEVICE:-cuda}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
echo "=========================================="
echo "Launching Aggregated Serving + LMCache (1 GPU)"
echo "=========================================="
echo "Model:       $MODEL"
echo "Frontend:    http://localhost:$HTTP_PORT"
echo "=========================================="
echo ""
echo "Example test command:"
echo ""
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"${MODEL}\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": \"Explain why Roger Federer is considered one of the greatest tennis players of all time\"}],"
echo "      \"max_tokens\": 32"
echo "    }'"
echo ""
echo "=========================================="

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with LMCache enabled (without PROMETHEUS_MULTIPROC_DIR set externally)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm --model "$MODEL" \
    "${BLOCK_SIZE_ARG[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\"}"
