#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run aiperf against the 8-frontend KV router topology via LB proxy.
#
# Usage:
#   bash benchmarks/multimodal/bench_8fe_kv.sh

MODEL="${DYN_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
LB_PORT="${DYN_LB_PORT:-9000}"
ARTIFACT_DIR="${DYN_ARTIFACT_DIR:-/dynamo-tmp/logs/04-15/8fe_kv}"
INPUT_FILE="${DYN_INPUT_FILE:-/workspace/benchmarks/multimodal/jsonl/1000req_2img_1600pool_400word_base64.jsonl}"

CONCURRENCY="${DYN_CONCURRENCY:-64}"
REQUEST_COUNT="${DYN_REQUEST_COUNT:-1000}"

echo "=== aiperf benchmark ==="
echo "Model:       $MODEL"
echo "Target:      http://localhost:$LB_PORT"
echo "Concurrency: $CONCURRENCY"
echo "Requests:    $REQUEST_COUNT"
echo "Input:       $INPUT_FILE"
echo "Artifacts:   $ARTIFACT_DIR"
echo ""

mkdir -p "$ARTIFACT_DIR"

aiperf profile \
  --model "$MODEL" \
  --url "http://localhost:${LB_PORT}/v1/chat/completions" \
  --endpoint-type chat \
  --streaming \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count 16 \
  --warmup-concurrency 4 \
  --input-file "$INPUT_FILE" \
  --custom-dataset-type single_turn \
  --extra-inputs "max_tokens:150" \
  --extra-inputs "min_tokens:150" \
  --extra-inputs "ignore_eos:true" \
  --extra-inputs "stream:true" \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui none
