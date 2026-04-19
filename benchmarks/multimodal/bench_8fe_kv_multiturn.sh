#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run aiperf multi-turn pinassistant workload against the 8-FE KV router topology.
# Requires launch_8fe_8w_kv.sh running in a separate window.
#
# Prerequisites:
#   1. aiperf with raw_payload support:
#      pip install --no-deps "git+https://github.com/ai-dynamo/aiperf.git@ajc/raw-payload-support"
#
#   2. Generate multi-turn dataset (from PR #7883 + #8163):
#      cd benchmarks/multimodal/jsonl/raw_replay
#      python generate_raw_replay.py \
#        --config pinassistant.yaml \
#        --num-conversations 2000 \
#        --output-dir /dynamo-tmp/data/multiturn/2000conv \
#        --image-mode base64 --seed 42
#
# Usage:
#   bash benchmarks/multimodal/bench_8fe_kv_multiturn.sh
#
# NOTE: --url must be BASE URL only. --endpoint-type chat appends
#       /v1/chat/completions automatically. Including the path in --url
#       causes double-pathing and 404 errors.

MODEL="${DYN_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
LB_PORT="${DYN_LB_PORT:-9000}"
ARTIFACT_DIR="${DYN_ARTIFACT_DIR:-/dynamo-tmp/logs/$(date +%m-%d)/8fe_kv_multiturn}"
INPUT_DIR="${DYN_INPUT_DIR:-/dynamo-tmp/data/multiturn/2000conv}"

REQUEST_RATE="${DYN_REQUEST_RATE:-32}"
DURATION="${DYN_DURATION:-120}"
GRACE="${DYN_GRACE:-30}"
WARMUP="${DYN_WARMUP:-32}"

# Large multi-turn datasets (18GB+) need extended configure timeout
CONFIGURE_TIMEOUT="${DYN_CONFIGURE_TIMEOUT:-1200}"

echo "=== aiperf multi-turn benchmark ==="
echo "Model:        $MODEL"
echo "Target:       http://localhost:$LB_PORT"
echo "Request rate: $REQUEST_RATE req/s (constant)"
echo "Duration:     ${DURATION}s + ${GRACE}s grace"
echo "Input:        $INPUT_DIR"
echo "Artifacts:    $ARTIFACT_DIR"
echo ""

mkdir -p "$ARTIFACT_DIR"

AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="$CONFIGURE_TIMEOUT" \
AIPERF_DATASET_CONFIGURATION_TIMEOUT="$CONFIGURE_TIMEOUT" \
aiperf profile \
  --model "$MODEL" \
  --url "http://localhost:${LB_PORT}" \
  --endpoint-type chat \
  --streaming \
  --request-rate "$REQUEST_RATE" \
  --request-rate-mode constant \
  --benchmark-duration "$DURATION" \
  --benchmark-grace-period "$GRACE" \
  --warmup-request-count "$WARMUP" \
  --warmup-concurrency 4 \
  --input-file "$INPUT_DIR" \
  --custom-dataset-type raw_payload \
  --tokenizer "$MODEL" \
  --tokenizer-trust-remote-code \
  --random-seed 42 \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui none
