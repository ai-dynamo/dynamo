#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DYNAMO_ROOT="${DYNAMO_ROOT:-/tmp/dynamo-continuum-kv-hints-experiment-rebased}"
VLLM_ROOT="${VLLM_ROOT:-/tmp/vllm-kv-hints-g1-actions-rebased}"
AIPERF_ROOT="${AIPERF_ROOT:-/localhome/local-karenc/aiperf}"
RUNTIME_VENV="${RUNTIME_VENV:-/localhome/local-karenc/vllm/.venv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_DIR="${MODEL_DIR:-/localhome/local-karenc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
FIXTURE="${FIXTURE:-/tmp/continuum-weka-correctness/fixture/trace.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/continuum-weka-correctness/networked}"
HTTP_PORT="${HTTP_PORT:-8194}"
SYSTEM_PORT="${SYSTEM_PORT:-8094}"
KV_EVENT_PORT="${KV_EVENT_PORT:-20084}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VLLM_EXTENSION_SOURCE="${VLLM_EXTENSION_SOURCE:-/localhome/local-karenc/vllm/vllm}"

mkdir -p "$OUTPUT_DIR"

export PATH="$RUNTIME_VENV/bin:$PATH"
export PYTHONPATH="$DYNAMO_ROOT/components/src:$DYNAMO_ROOT/lib/bindings/python/src:$VLLM_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
export DYN_LOG=info
export DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY=1
export DYN_EXPERIMENTAL_SESSION_KV_HINT_RETENTION_PRIORITY=10
export DYN_EXPERIMENTAL_SESSION_KV_HINT_RETENTION_TTL_SECONDS=300

frontend_pid=""
worker_pid=""
extension_links=()

cleanup() {
  if [[ -n "$worker_pid" ]]; then
    kill "$worker_pid" 2>/dev/null || true
  fi
  if [[ -n "$frontend_pid" ]]; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  wait "$worker_pid" "$frontend_pid" 2>/dev/null || true
  if ((${#extension_links[@]})); then
    rm -f "${extension_links[@]}"
  fi
}
trap cleanup EXIT

while IFS= read -r relative_path; do
  target="$VLLM_ROOT/vllm/$relative_path"
  if [[ ! -e "$target" ]]; then
    mkdir -p "$(dirname "$target")"
    ln -s "$VLLM_EXTENSION_SOURCE/$relative_path" "$target"
    extension_links+=("$target")
  fi
done < <(cd "$VLLM_EXTENSION_SOURCE" && find . -type f -name '*.so' -printf '%P\n')

python -m dynamo.frontend \
  --router-mode kv \
  --enable-session-prefix-index \
  --http-port "$HTTP_PORT" \
  >"$OUTPUT_DIR/frontend.log" 2>&1 &
frontend_pid=$!

kv_events_config=$(printf '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%s","enable_kv_cache_events":true}' "$KV_EVENT_PORT")
DYN_SYSTEM_PORT="$SYSTEM_PORT" CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  python -m dynamo.vllm \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --block-size 64 \
  --max-model-len 40960 \
  --max-num-seqs 1 \
  --num-gpu-blocks-override 1024 \
  --gpu-memory-utilization 0.5 \
  --enable-prefix-caching \
  --enforce-eager \
  --kv-events-config "$kv_events_config" \
  >"$OUTPUT_DIR/worker.log" 2>&1 &
worker_pid=$!

for _ in $(seq 1 240); do
  if ! kill -0 "$frontend_pid" 2>/dev/null; then
    echo "Dynamo frontend exited before becoming ready" >&2
    exit 1
  fi
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    echo "vLLM worker exited before becoming ready" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$HTTP_PORT/v1/models" | grep -q "$MODEL_NAME"; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$HTTP_PORT/v1/models" | grep -q "$MODEL_NAME"; then
  echo "Model did not register before timeout" >&2
  exit 1
fi

AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true \
  "$AIPERF_ROOT/.venv/bin/aiperf" profile \
  --scenario inferencex-agentx-mvp \
  --model "$MODEL_NAME" \
  --url "http://127.0.0.1:$HTTP_PORT" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --custom-dataset-type weka_trace \
  --input-file "$FIXTURE" \
  --num-dataset-entries 1 \
  --concurrency 1 \
  --request-count 13 \
  --benchmark-duration 180 \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "$MODEL_DIR" \
  --tokenizer-trust-remote-code \
  --slice-duration 1.0 \
  --unsafe-override \
  --output-artifact-dir "$OUTPUT_DIR/aiperf" \
  >"$OUTPUT_DIR/aiperf.log" 2>&1

sleep 5
grep 'continuum_kv_hints' "$OUTPUT_DIR/frontend.log" >"$OUTPUT_DIR/policy-actions.log" || true
echo "Results written to $OUTPUT_DIR"
