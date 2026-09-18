#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

image="${CONTINUUM_IMAGE:?Set CONTINUUM_IMAGE to the pinned experiment image}"
aiperf_root="${AIPERF_ROOT:-/localhome/local-karenc/aiperf}"
fixture="${FIXTURE:-/tmp/continuum-weka-correctness/fixture/trace.json}"
output_dir="${OUTPUT_DIR:-/tmp/continuum-weka-correctness/container-networked}"
gpu_device="${GPU_DEVICE:-1}"
http_port="${HTTP_PORT:-8195}"
system_port="${SYSTEM_PORT:-8095}"
kv_event_port="${KV_EVENT_PORT:-20085}"
model_repo="${MODEL_REPO:-/localhome/local-karenc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B}"
model_dir="${MODEL_DIR:-$model_repo/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
model_snapshot="/models/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
container="continuum-weka-$$"

mkdir -p "$output_dir"
chmod 0777 "$output_dir"

cleanup() {
  docker stop "$container" >/dev/null 2>&1 || true
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d \
  --name "$container" \
  --gpus "device=$gpu_device" \
  --network host \
  --ipc host \
  -v "$model_repo:/models/Qwen3-0.6B:ro" \
  -v "$output_dir:/results" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONHASHSEED=0 \
  -e DYN_LOG=info \
  -e DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY=1 \
  -e DYN_EXPERIMENTAL_SESSION_KV_HINT_RETENTION_PRIORITY=10 \
  -e DYN_EXPERIMENTAL_SESSION_KV_HINT_RETENTION_TTL_SECONDS=300 \
  "$image" \
  bash -lc "set -euo pipefail; \
    python -m dynamo.frontend \
      --router-mode kv \
      --enable-session-prefix-index \
      --http-port '$http_port' \
      > /results/frontend.log 2>&1 & \
    frontend_pid=\$!; \
    DYN_SYSTEM_PORT='$system_port' CUDA_VISIBLE_DEVICES=0 \
      python -m dynamo.vllm \
        --model '$model_snapshot' \
        --served-model-name Qwen/Qwen3-0.6B \
        --block-size 64 \
        --max-model-len 40960 \
        --max-num-seqs 1 \
        --num-gpu-blocks-override 1024 \
        --gpu-memory-utilization 0.5 \
        --enable-prefix-caching \
        --enforce-eager \
        --kv-events-config '{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$kv_event_port\",\"enable_kv_cache_events\":true}' \
        > /results/worker.log 2>&1 & \
    worker_pid=\$!; \
    trap 'kill \"\$worker_pid\" \"\$frontend_pid\" 2>/dev/null || true' EXIT TERM INT; \
    wait -n \"\$frontend_pid\" \"\$worker_pid\"" \
  >"$output_dir/container-id.txt"

for _ in $(seq 1 240); do
  if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
    echo "container exited before readiness" >&2
    exit 1
  fi
  if rg -q 'has been initialized' "$output_dir/worker.log" 2>/dev/null \
    && curl -fsS "http://127.0.0.1:$http_port/v1/models" | grep -q 'Qwen/Qwen3-0.6B'; then
    break
  fi
  sleep 1
done

AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true \
  "$aiperf_root/.venv/bin/aiperf" profile \
  --scenario inferencex-agentx-mvp \
  --model Qwen/Qwen3-0.6B \
  --url "http://127.0.0.1:$http_port" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --custom-dataset-type weka_trace \
  --input-file "$fixture" \
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
  --tokenizer "$model_dir" \
  --tokenizer-trust-remote-code \
  --slice-duration 1.0 \
  --unsafe-override \
  --output-artifact-dir "$output_dir/aiperf" \
  >"$output_dir/aiperf.log" 2>&1

sleep 5
sed -E 's/\x1B\[[0-9;]*[mK]//g' "$output_dir/frontend.log" \
  | rg 'continuum_kv_hints.*Emitting request-completion KV hint' \
  >"$output_dir/policy-actions.log" || true
