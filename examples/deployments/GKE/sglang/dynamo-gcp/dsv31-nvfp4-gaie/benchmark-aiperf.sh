#!/usr/bin/env bash
# NVFP4 GAIE benchmarks via Inference Gateway + aiperf
#
# Run inside a pod that has `aiperf` and tokenizer cache (e.g. perf-gaie-fp8):
#   GATEWAY_IP=<lb-ip> CONCURRENCY=100 X_POOL=gaie-agg bash benchmark-aiperf.sh
#
# Native Dynamo (no Gateway / no x-pool): call aiperf with --url pointing at the
# frontend Service and drop --header. Examples:
#   --url 'http://sglang-dsv31-nvfp4-agg-native-frontend.dynamo-system.svc.cluster.local:8000'
#   --url 'http://sglang-dsv31-nvfp4-disagg-native-frontend.dynamo-system.svc.cluster.local:8000'
#
# Prerequisites:
#   - Gateway address (LB IP). From a host with kubectl:
#       kubectl get gateway inference-gateway -n dynamo-system -o jsonpath='{.status.addresses[0].value}{"\n"}'
#   - From inside the cluster you can use the Gateway Service ClusterIP instead of the LB IP.

set -euo pipefail

# --- Gateway (set before running) ---
# shellcheck disable=SC2155
: "${GATEWAY_IP:?Set GATEWAY_IP to the inference-gateway external IP or reachable address}"

# --- Which GAIE pool to hit (HTTP header sent to the Gateway) ---
#   gaie-disagg-nvfp4  -> NVFP4 disaggregated GAIE (x-pool from httproute.yaml)
#   gaie-agg           -> NVFP4 aggregated GAIE (default route also targets agg pool)
X_POOL="${X_POOL:-gaie-disagg-nvfp4}"

# --- Concurrency sweep (edit CONCURRENCY per run) ---
# This is the main knob for load: how many concurrent requests aiperf keeps in flight.
# Typical sweep for crossover / capacity studies: 10, 50, 100, 200, 500
# (workspace optimization notes often call out C=50 and C=100 explicitly).
#
# Also adjust (optional):
#   REQUEST_COUNT   — total requests to send (should be >= CONCURRENCY for meaningful overlap)
#   WORKERS_MAX     — aiperf client parallelism cap (often 50; raise if client becomes the bottleneck)
CONCURRENCY="${CONCURRENCY:-100}"
REQUEST_COUNT="${REQUEST_COUNT:-500}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-20}"
WORKERS_MAX="${WORKERS_MAX:-50}"

# --- Workload shape (ISL / OSL) ---
# Input ~1000 tokens, output ~250 tokens (matches README result tables).
INPUT_TOKENS_MEAN="${INPUT_TOKENS_MEAN:-1000}"
OUTPUT_TOKENS_MEAN="${OUTPUT_TOKENS_MEAN:-250}"

# --- Artifacts ---
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/results/nvfp4}"
# shellcheck disable=SC2154
SAFE_POOL="${X_POOL//[^a-zA-Z0-9_-]/_}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ARTIFACT_ROOT}/gaie-${SAFE_POOL}-c${CONCURRENCY}}"

# --- Model / tokenizer (paths on perf pod) ---
MODEL="${MODEL:-nvidia/DeepSeek-V3.1-NVFP4}"
TOKENIZER="${TOKENIZER:-/opt/model-cache/hub/models--nvidia--DeepSeek-V3.1-NVFP4/snapshots/68b4a17cce1482e94030ca00dacda3dec4c6359d}"

echo "GATEWAY_IP=$GATEWAY_IP  X_POOL=$X_POOL  CONCURRENCY=$CONCURRENCY  ARTIFACT_DIR=$ARTIFACT_DIR"

mkdir -p "$ARTIFACT_DIR"

exec aiperf profile \
  --url "http://${GATEWAY_IP}" \
  --header "x-pool:${X_POOL}" \
  --artifact-dir "$ARTIFACT_DIR" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --streaming \
  --synthetic-input-tokens-mean "$INPUT_TOKENS_MEAN" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OUTPUT_TOKENS_MEAN" \
  --output-tokens-stddev 0 \
  --extra-inputs "max_tokens:${OUTPUT_TOKENS_MEAN}" \
  --extra-inputs "min_tokens:${OUTPUT_TOKENS_MEAN}" \
  --extra-inputs ignore_eos:true \
  --extra-inputs repetition_penalty:1.0 \
  --extra-inputs temperature:0.0 \
  --use-server-token-count \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count "$WARMUP_REQUESTS" \
  --num-dataset-entries 12800 \
  --random-seed 100 \
  --workers-max "$WORKERS_MAX" \
  --record-processors 16
