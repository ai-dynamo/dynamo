#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Benchmark: Plain SGLang vs Dynamo.SGLang
#
# Runs aiperf benchmarks against both plain sglang and dynamo.sglang using the
# same model, configuration, and aiperf flags for a fair A/B comparison.
#
# Designed to run inside the sglang-runtime container:
#   nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0-dev.2
#
# Usage:
#   ./sglang_vs_dynamo_sglang.sh [OPTIONS]
#
# Example (Qwen3.5-0.8B with defaults):
#   ./sglang_vs_dynamo_sglang.sh
#
# Example (custom model and concurrency):
#   ./sglang_vs_dynamo_sglang.sh --model Qwen/Qwen3-0.6B --concurrency 1,2,4
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (tuned for Qwen3.5-0.8B)
# ---------------------------------------------------------------------------
MODEL="Qwen/Qwen3.5-0.8B"
PORT=8000
PAGE_SIZE=1
TP=1
ISL=128
OSL=128
CONCURRENCY_LIST="1,2,4,8,16"
ARTIFACTS_DIR="artifacts_sglang_vs_dynamo"
HEALTH_TIMEOUT=300   # seconds to wait for server readiness
EXTRA_SGLANG_ARGS=""

print_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Benchmark plain SGLang vs Dynamo.SGLang with identical aiperf flags.

Options:
  --model <model_id>         HuggingFace model ID (default: $MODEL)
  --port <int>               HTTP port for the server (default: $PORT)
  --page-size <int>          KV cache page size (default: $PAGE_SIZE)
  --tp <int>                 Tensor parallelism (default: $TP)
  --isl <int>                Input sequence length (default: $ISL)
  --osl <int>                Output sequence length (default: $OSL)
  --concurrency <list>       Comma-separated concurrency levels (default: $CONCURRENCY_LIST)
  --artifacts-dir <path>     Directory for benchmark results (default: $ARTIFACTS_DIR)
  --health-timeout <int>     Seconds to wait for server health (default: $HEALTH_TIMEOUT)
  --extra-args <string>      Extra args passed to both sglang and dynamo.sglang
  --help                     Show this help message
EOF
  exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)           MODEL="$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --page-size)       PAGE_SIZE="$2"; shift 2 ;;
    --tp)              TP="$2"; shift 2 ;;
    --isl)             ISL="$2"; shift 2 ;;
    --osl)             OSL="$2"; shift 2 ;;
    --concurrency)     CONCURRENCY_LIST="$2"; shift 2 ;;
    --artifacts-dir)   ARTIFACTS_DIR="$2"; shift 2 ;;
    --health-timeout)  HEALTH_TIMEOUT="$2"; shift 2 ;;
    --extra-args)      EXTRA_SGLANG_ARGS="$2"; shift 2 ;;
    --help)            print_help ;;
    *)                 echo "Unknown option: $1"; exit 1 ;;
  esac
done

IFS=',' read -r -a CONCURRENCY_ARRAY <<< "$CONCURRENCY_LIST"

# Validate concurrency values
for val in "${CONCURRENCY_ARRAY[@]}"; do
  if ! [[ "$val" =~ ^[0-9]+$ ]] || [ "$val" -le 0 ]; then
    echo "Error: Invalid concurrency value '$val'. Must be a positive integer." >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v aiperf &>/dev/null; then
  echo "ERROR: aiperf not found. Install with: pip install git+https://github.com/ai-dynamo/aiperf.git"
  exit 1
fi

SGLANG_DIR="${ARTIFACTS_DIR}/sglang"
DYNAMO_DIR="${ARTIFACTS_DIR}/dynamo_sglang"
mkdir -p "$SGLANG_DIR" "$DYNAMO_DIR"

URL="http://localhost:${PORT}"

echo "=========================================="
echo "SGLang vs Dynamo.SGLang Benchmark"
echo "=========================================="
echo "Model:        $MODEL"
echo "Page size:    $PAGE_SIZE"
echo "TP:           $TP"
echo "ISL:          $ISL"
echo "OSL:          $OSL"
echo "Concurrency:  ${CONCURRENCY_ARRAY[*]}"
echo "Port:         $PORT"
echo "Artifacts:    $ARTIFACTS_DIR"
echo "=========================================="

# ---------------------------------------------------------------------------
# Helper: wait for OpenAI-compatible /v1/models endpoint to report the model
# ---------------------------------------------------------------------------
wait_for_model() {
  local url="$1"
  local model="$2"
  local timeout="$3"
  local waited=0

  echo "Waiting for model '$model' at ${url}/v1/models (timeout: ${timeout}s)..."
  while ! curl -sf "${url}/v1/models" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
sys.exit(0 if '$model' in models else 1)
" 2>/dev/null; do
    if [ "$waited" -ge "$timeout" ]; then
      echo "ERROR: Timed out waiting for model after ${timeout}s"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 30)) -eq 0 ]; then
      echo "  Still waiting... (${waited}s / ${timeout}s)"
    fi
  done
  echo "Model ready after ${waited}s."
}

# ---------------------------------------------------------------------------
# Helper: run aiperf for all concurrency levels
# ---------------------------------------------------------------------------
run_aiperf_sweep() {
  local artifact_base="$1"
  local deployment_kind="$2"

  for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    local artifact_dir="${artifact_base}/concurrency${concurrency}"
    mkdir -p "$artifact_dir"
    echo ""
    echo "--- aiperf: ${deployment_kind} @ concurrency=${concurrency} ---"

    # NOTE: For Dynamo HTTP OpenAI frontend, use `nvext` for fields like
    # `ignore_eos` since they are not in the official OpenAI spec.
    aiperf profile \
      --model "${MODEL}" \
      --tokenizer "${MODEL}" \
      --endpoint-type chat \
      --endpoint /v1/chat/completions \
      --streaming \
      --url "${URL}" \
      --synthetic-input-tokens-mean "${ISL}" \
      --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean "${OSL}" \
      --output-tokens-stddev 0 \
      --extra-inputs "max_tokens:${OSL}" \
      --extra-inputs "min_tokens:${OSL}" \
      --extra-inputs "ignore_eos:true" \
      --extra-inputs '{"nvext":{"ignore_eos":true}}' \
      --concurrency "${concurrency}" \
      --request-count $((concurrency * 10)) \
      --warmup-request-count $((concurrency * 2)) \
      --num-dataset-entries $((concurrency * 12)) \
      --random-seed 100 \
      --artifact-dir "${artifact_dir}" \
      --ui simple \
      -H 'Authorization: Bearer NOT USED' \
      -H 'Accept: text/event-stream'
  done

  # Save deployment config
  cat > "${artifact_base}/deployment_config.json" <<EOF
{
  "kind": "${deployment_kind}",
  "model": "${MODEL}",
  "input_sequence_length": ${ISL},
  "output_sequence_length": ${OSL},
  "tensor_parallelism": ${TP},
  "page_size": ${PAGE_SIZE}
}
EOF
}

# ---------------------------------------------------------------------------
# Helper: kill server process group
# ---------------------------------------------------------------------------
cleanup_server() {
  local pids="$1"
  echo "Stopping server (PIDs: ${pids})..."
  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
  done
  # Wait briefly for graceful shutdown
  sleep 3
  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Server stopped."
}

# =====================================================================
# Phase 1: Benchmark plain SGLang
# =====================================================================
echo ""
echo "=========================================="
echo "Phase 1: Plain SGLang"
echo "=========================================="

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --page-size "$PAGE_SIZE" \
  --tp "$TP" \
  --trust-remote-code \
  $EXTRA_SGLANG_ARGS &
SGLANG_PID=$!
echo "Started plain sglang (PID: $SGLANG_PID)"

if ! wait_for_model "$URL" "$MODEL" "$HEALTH_TIMEOUT"; then
  echo "ERROR: Plain sglang failed to start."
  cleanup_server "$SGLANG_PID"
  exit 1
fi

run_aiperf_sweep "$SGLANG_DIR" "sglang"

cleanup_server "$SGLANG_PID"

# =====================================================================
# Phase 2: Benchmark Dynamo.SGLang
# =====================================================================
echo ""
echo "=========================================="
echo "Phase 2: Dynamo.SGLang"
echo "=========================================="

# Launch dynamo frontend
DYN_HTTP_PORT="$PORT" python3 -m dynamo.frontend &
FRONTEND_PID=$!
echo "Started dynamo.frontend (PID: $FRONTEND_PID)"

# Launch dynamo.sglang worker
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size "$PAGE_SIZE" \
  --tp "$TP" \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-metrics \
  $EXTRA_SGLANG_ARGS &
WORKER_PID=$!
echo "Started dynamo.sglang worker (PID: $WORKER_PID)"

if ! wait_for_model "$URL" "$MODEL" "$HEALTH_TIMEOUT"; then
  echo "ERROR: Dynamo.sglang failed to start."
  cleanup_server "$FRONTEND_PID $WORKER_PID"
  exit 1
fi

run_aiperf_sweep "$DYNAMO_DIR" "dynamo_sglang"

cleanup_server "$FRONTEND_PID $WORKER_PID"

# =====================================================================
# Phase 3: Compare results
# =====================================================================
echo ""
echo "=========================================="
echo "Phase 3: Comparing Results"
echo "=========================================="

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
python3 "${SCRIPT_DIR}/compare_results.py" \
  --baseline "$SGLANG_DIR" \
  --candidate "$DYNAMO_DIR" \
  --baseline-name "sglang" \
  --candidate-name "dynamo.sglang" \
  --threshold 5.0

echo ""
echo "=========================================="
echo "Benchmark complete. Results in: $ARTIFACTS_DIR"
echo "=========================================="
