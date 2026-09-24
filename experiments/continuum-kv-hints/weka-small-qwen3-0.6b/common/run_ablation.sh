#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

case_name="${1:?usage: run_ablation.sh CASE POLICY_MODE}"
policy_mode="${2:?usage: run_ablation.sh CASE POLICY_MODE}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"
dynamo_root="${DYNAMO_ROOT:-$(git -C "$experiment_dir" rev-parse --show-toplevel)}"
aiperf_root="${AIPERF_ROOT:-/localhome/local-karenc/aiperf}"
aiperf_bin="${AIPERF_BIN:-$aiperf_root/.venv/bin/aiperf}"
aiperf_python="${AIPERF_PYTHON:-$aiperf_root/.venv/bin/python}"
image="${CONTINUUM_IMAGE:?Set CONTINUUM_IMAGE to the local experiment image}"
model_name="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
model_repo="${MODEL_REPO:-/localhome/local-karenc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B}"
model_revision="${MODEL_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}"
model_host_path="$model_repo/snapshots/$model_revision"
model_container_path="/models/Qwen3-0.6B/snapshots/$model_revision"
hf_dataset="${HF_DATASET:-semianalysisai/cc-traces-weka-062126-256k}"
num_traces="${NUM_TRACES:-4}"
max_context_length="${MAX_CONTEXT_LENGTH:-32768}"
max_requests_per_trace="${MAX_REQUESTS_PER_TRACE:-120}"
trace_selection_mode="${TRACE_SELECTION_MODE:-source-order}"
max_source_span_seconds="${MAX_SOURCE_SPAN_SECONDS:-}"
token_scale_factor="${TOKEN_SCALE_FACTOR:-8}"
time_scale_factor="${TIME_SCALE_FACTOR:-0.01}"
max_think_time="${MAX_THINK_TIME:-2.0}"
concurrency="${CONCURRENCY:-4}"
benchmark_duration="${BENCHMARK_DURATION:-300}"
request_count="${REQUEST_COUNT:-}"
num_sessions="${NUM_SESSIONS:-}"
trajectory_start_min_ratio="${TRAJECTORY_START_MIN_RATIO:-0.25}"
trajectory_start_max_ratio="${TRAJECTORY_START_MAX_RATIO:-0.75}"
system_idle_gap_cap_seconds="${SYSTEM_IDLE_GAP_CAP_SECONDS:-}"
gpu_blocks="${GPU_BLOCKS:-1024}"
gpu_device="${GPU_DEVICE:-1}"
retention_priority="${RETENTION_PRIORITY:-10}"
retention_max_fraction="${RETENTION_MAX_FRACTION:-0.25}"
inferred_tool_min_think_time_seconds="${INFERRED_TOOL_MIN_THINK_TIME_SECONDS:-}"
retention_oracle_schedule="${RETENTION_ORACLE_SCHEDULE:-}"
capture_raw_kv_events="${CAPTURE_RAW_KV_EVENTS:-0}"
require_idle_gpu="${REQUIRE_IDLE_GPU:-0}"
http_port="${HTTP_PORT:-8194}"
system_port="${SYSTEM_PORT:-8094}"
kv_event_port="${KV_EVENT_PORT:-20084}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${OUTPUT_ROOT:-$experiment_dir/artifacts/$run_id}"
output_dir="$output_root/$case_name"
dataset_dir="$output_root/generated-dataset"
container="continuum-weka-${case_name//[^a-zA-Z0-9_.-]/-}-$$"
namespace="continuum-weka-${run_id}-${case_name}"
raw_kv_capture_pid=""

test -x "$aiperf_bin"
test -x "$aiperf_python"
test -d "$model_host_path"
mkdir -p "$output_dir/aiperf"
chmod 0777 "$output_dir" "$output_dir/aiperf"

capture_gpu_state() {
  local output="$1"
  {
    nvidia-smi -i "$gpu_device" \
      --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw \
      --format=csv,noheader,nounits
    nvidia-smi -i "$gpu_device" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits || true
  } >"$output"
}

capture_gpu_state "$output_dir/gpu-state-before.txt"
if [[ "$require_idle_gpu" == "1" ]]; then
  compute_apps="$({
    nvidia-smi -i "$gpu_device" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits || true
  } | sed '/^[[:space:]]*$/d')"
  if [[ -n "$compute_apps" ]]; then
    printf 'GPU %s has active compute processes:\n%s\n' \
      "$gpu_device" "$compute_apps" >&2
    exit 1
  fi
fi

policy_env=()
case "$policy_mode" in
  none)
    ;;
  retain | inferred-tool-retain | all-retain | combined | all-retain-and-evict)
    policy_env+=(
      -e DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY=1
      -e "DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY_MODE=$policy_mode"
      -e "DYN_EXPERIMENTAL_SESSION_KV_HINT_RETENTION_PRIORITY=$retention_priority"
    )
    ;;
  evict)
    policy_env+=(
      -e DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY=1
      -e DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY_MODE=evict
    )
    ;;
  *)
    echo "unsupported policy mode: $policy_mode" >&2
    exit 2
    ;;
esac

case "$policy_mode" in
  inferred-tool-retain | all-retain | all-retain-and-evict)
    if [[ -z "$inferred_tool_min_think_time_seconds" \
      && -z "$retention_oracle_schedule" ]]; then
      echo "Set INFERRED_TOOL_MIN_THINK_TIME_SECONDS or RETENTION_ORACLE_SCHEDULE for $policy_mode" >&2
      exit 2
    fi
    ;;
esac

cleanup() {
  if [[ -n "$raw_kv_capture_pid" ]]; then
    kill "$raw_kv_capture_pid" >/dev/null 2>&1 || true
    wait "$raw_kv_capture_pid" 2>/dev/null || true
  fi
  docker stop "$container" >/dev/null 2>&1 || true
  docker rm "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  printf 'run_id=%s\n' "$run_id"
  printf 'case=%s\n' "$case_name"
  printf 'policy_mode=%s\n' "$policy_mode"
  printf 'dynamo_commit=%s\n' "$(git -C "$dynamo_root" rev-parse HEAD)"
  printf 'dynamo_dirty=%s\n' "$(test -n "$(git -C "$dynamo_root" status --porcelain)" && echo true || echo false)"
  printf 'aiperf_commit=%s\n' "$(git -C "$aiperf_root" rev-parse HEAD)"
  printf 'aiperf_dirty=%s\n' "$(test -n "$(git -C "$aiperf_root" status --porcelain)" && echo true || echo false)"
  printf 'image=%s\n' "$image"
  printf 'image_id=%s\n' "$(docker image inspect "$image" --format '{{.Id}}')"
  printf 'model=%s\n' "$model_name"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'hf_dataset=%s\n' "$hf_dataset"
  printf 'num_traces=%s\n' "$num_traces"
  printf 'max_context_length=%s\n' "$max_context_length"
  printf 'max_requests_per_trace=%s\n' "$max_requests_per_trace"
  printf 'trace_selection_mode=%s\n' "$trace_selection_mode"
  printf 'max_source_span_seconds=%s\n' "$max_source_span_seconds"
  printf 'token_scale_factor=%s\n' "$token_scale_factor"
  printf 'time_scale_factor=%s\n' "$time_scale_factor"
  printf 'max_think_time=%s\n' "$max_think_time"
  printf 'concurrency=%s\n' "$concurrency"
  printf 'benchmark_duration=%s\n' "$benchmark_duration"
  printf 'request_count=%s\n' "$request_count"
  printf 'num_sessions=%s\n' "$num_sessions"
  printf 'trajectory_start_min_ratio=%s\n' "$trajectory_start_min_ratio"
  printf 'trajectory_start_max_ratio=%s\n' "$trajectory_start_max_ratio"
  printf 'system_idle_gap_cap_seconds=%s\n' "$system_idle_gap_cap_seconds"
  printf 'gpu_blocks=%s\n' "$gpu_blocks"
  printf 'gpu_device=%s\n' "$gpu_device"
  printf 'retention_priority=%s\n' "$retention_priority"
  printf 'retention_max_fraction=%s\n' "$retention_max_fraction"
  printf 'inferred_tool_min_think_time_seconds=%s\n' \
    "$inferred_tool_min_think_time_seconds"
  printf 'retention_oracle_schedule=%s\n' "$retention_oracle_schedule"
  printf 'capture_raw_kv_events=%s\n' "$capture_raw_kv_events"
  printf 'require_idle_gpu=%s\n' "$require_idle_gpu"
} >"$output_dir/run.env"
git -C "$dynamo_root" diff --binary >"$output_dir/dynamo-working-tree.patch"
git -C "$aiperf_root" diff --binary >"$output_dir/aiperf-working-tree.patch"

if [[ ! -f "$dataset_dir/manifest.json" ]]; then
  dataset_command=(
    "$aiperf_python" "$script_dir/prepare_dataset.py"
    --dataset "$hf_dataset"
    --num-traces "$num_traces"
    --max-requests-per-trace "$max_requests_per_trace"
    --selection-mode "$trace_selection_mode"
    --token-scale-factor "$token_scale_factor"
    --time-scale-factor "$time_scale_factor"
    --max-think-time "$max_think_time"
    --max-context-length "$max_context_length"
    --output-dir "$dataset_dir"
  )
  if [[ -n "$max_source_span_seconds" ]]; then
    dataset_command+=(--max-source-span-seconds "$max_source_span_seconds")
  fi
  printf '%q ' "${dataset_command[@]}" >"$output_root/dataset-preparation-command.txt"
  printf '\n' >>"$output_root/dataset-preparation-command.txt"
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH="$aiperf_root/src" \
    "${dataset_command[@]}" >"$output_root/dataset-preparation.log" 2>&1
fi
cp "$dataset_dir/manifest.json" "$output_dir/subset-manifest.json"

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
  -e "DYN_NAMESPACE=$namespace" \
  "${policy_env[@]}" \
  "$image" \
  bash -lc "set -euo pipefail; \
    if [[ -n \"\${PROMETHEUS_MULTIPROC_DIR:-}\" ]]; then \
      rm -rf \"\$PROMETHEUS_MULTIPROC_DIR\"; \
      mkdir -p \"\$PROMETHEUS_MULTIPROC_DIR\"; \
    fi; \
    python -m dynamo.frontend \
      --router-mode kv \
      --enable-session-prefix-index \
      --http-port '$http_port' \
      > /results/frontend.log 2>&1 & \
    frontend_pid=\$!; \
    DYN_SYSTEM_PORT='$system_port' CUDA_VISIBLE_DEVICES=0 \
      python -m dynamo.vllm \
        --model '$model_container_path' \
        --served-model-name '$model_name' \
        --block-size 64 \
        --max-model-len 40960 \
        --max-num-seqs 8 \
        --num-gpu-blocks-override '$gpu_blocks' \
        --gpu-memory-utilization 0.5 \
        --kv-cache-retention-max-fraction '$retention_max_fraction' \
        --enable-prefix-caching \
        --enforce-eager \
        --kv-events-config '{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$kv_event_port\",\"enable_kv_cache_events\":true}' \
        > /results/worker.log 2>&1 & \
    worker_pid=\$!; \
    trap 'kill \"\$worker_pid\" \"\$frontend_pid\" 2>/dev/null || true' EXIT TERM INT; \
    wait -n \"\$frontend_pid\" \"\$worker_pid\"" \
  >"$output_dir/container-id.txt"

for _ in $(seq 1 300); do
  if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
    echo "Dynamo vLLM container exited before readiness" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$http_port/v1/models" | grep -q "$model_name"; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$http_port/v1/models" | grep -q "$model_name"; then
  echo "Dynamo vLLM did not become ready" >&2
  exit 1
fi

if [[ "$capture_raw_kv_events" == "1" ]]; then
  "$aiperf_python" "$script_dir/capture_raw_kv_events.py" \
    --endpoint "tcp://127.0.0.1:$kv_event_port" \
    --output "$output_dir/raw-kv-events.jsonl" &
  raw_kv_capture_pid=$!
  sleep 1
fi

aiperf_command=(
  "$aiperf_bin" profile
  --scenario inferencex-agentx-mvp
  --url "http://127.0.0.1:$http_port"
  --endpoint /v1/chat/completions
  --endpoint-type chat
  --streaming
  --model "$model_name"
  --custom-dataset-type weka_trace
  --input-file "$dataset_dir/traces"
  --max-context-length "$max_context_length"
  --num-dataset-entries "$num_traces"
  --concurrency "$concurrency"
  --benchmark-duration "$benchmark_duration"
  --stats-interval 30
  --random-seed 42
  --failed-request-threshold 0.10
  --trajectory-start-min-ratio "$trajectory_start_min_ratio"
  --trajectory-start-max-ratio "$trajectory_start_max_ratio"
  --use-server-token-count
  --no-gpu-telemetry
  --tokenizer "$model_host_path"
  --tokenizer-trust-remote-code
  --slice-duration 1.0
  --unsafe-override
  --export-level records
  --profile-export-prefix profile
  --output-artifact-dir "$output_dir/aiperf"
  --server-metrics "http://127.0.0.1:$http_port/metrics"
)
if [[ -n "$system_idle_gap_cap_seconds" ]]; then
  aiperf_command+=(--system-idle-gap-cap-seconds "$system_idle_gap_cap_seconds")
fi
if [[ -n "$request_count" ]]; then
  aiperf_command+=(--request-count "$request_count")
fi
if [[ -n "$num_sessions" ]]; then
  aiperf_command+=(--num-conversations "$num_sessions")
fi
printf '%q ' "${aiperf_command[@]}" >"$output_dir/aiperf-command.txt"
printf '\n' >>"$output_dir/aiperf-command.txt"

date +%s.%N >"$output_dir/benchmark-start-epoch.txt"
set +e
aiperf_env=(
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  "PYTHONPATH=$aiperf_root/src${PYTHONPATH:+:$PYTHONPATH}"
)
if [[ -n "$inferred_tool_min_think_time_seconds" \
  || -n "$retention_oracle_schedule" ]]; then
  aiperf_env+=(
    AIPERF_DATASET_MMAP_CACHE_ENABLED=false
  )
fi
if [[ -n "$inferred_tool_min_think_time_seconds" ]]; then
  aiperf_env+=(
    "AIPERF_HTTP_DYNAMO_INFERRED_TOOL_CALL_MIN_THINK_TIME_SECONDS=$inferred_tool_min_think_time_seconds"
  )
fi
if [[ -n "$retention_oracle_schedule" ]]; then
  aiperf_env+=("AIPERF_DYNAMO_RETENTION_ORACLE_SCHEDULE=$retention_oracle_schedule")
fi
env "${aiperf_env[@]}" "${aiperf_command[@]}" >"$output_dir/aiperf.log" 2>&1
aiperf_status=$?
set -e
date +%s.%N >"$output_dir/benchmark-end-epoch.txt"

sleep 5
if [[ -n "$raw_kv_capture_pid" ]]; then
  kill "$raw_kv_capture_pid" 2>/dev/null || true
  wait "$raw_kv_capture_pid" 2>/dev/null || true
  raw_kv_capture_pid=""
fi
curl -fsS "http://127.0.0.1:$http_port/metrics" >"$output_dir/dynamo-metrics-final.prom" || true
capture_gpu_state "$output_dir/gpu-state-after-benchmark.txt"
sed -E 's/\x1B\[[0-9;]*[mK]//g' "$output_dir/frontend.log" \
  | rg 'continuum_kv_hints.*Emitting request-completion KV hint' \
  >"$output_dir/policy-actions.log" || true
rg -i 'BlockRemoved|BlockStored|block to remove|block to store' \
  "$output_dir/frontend.log" "$output_dir/worker.log" \
  >"$output_dir/kv-events.log" || true
printf '%s\n' "$aiperf_status" >"$output_dir/aiperf-exit-code.txt"
"$aiperf_python" "$script_dir/summarize_retention_occupancy.py" \
  --worker-log "$output_dir/worker.log" \
  --aiperf-log "$output_dir/aiperf.log" \
  --benchmark-start "$output_dir/benchmark-start-epoch.txt" \
  --benchmark-end "$output_dir/benchmark-end-epoch.txt" \
  --summary-output "$output_dir/retention-occupancy-summary.json" \
  --series-output "$output_dir/retention-occupancy.csv" \
  >"$output_dir/retention-occupancy.log"

if ((aiperf_status != 0)); then
  echo "AIPerf failed with status $aiperf_status; see $output_dir/aiperf.log" >&2
  exit "$aiperf_status"
fi

if [[ "$policy_mode" == "retain" || "$policy_mode" == "inferred-tool-retain" \
  || "$policy_mode" == "all-retain" || "$policy_mode" == "combined" \
  || "$policy_mode" == "all-retain-and-evict" ]]; then
  if ! rg -q 'action_type="kv\.retain"' "$output_dir/policy-actions.log"; then
    echo "Oracle run emitted no kv.retain actions" >&2
    exit 1
  fi
  if rg 'action_type="kv\.retain"' "$output_dir/policy-actions.log" \
    | rg -qv 'ttl_source="request"'; then
    echo "Oracle run emitted a kv.retain action without a request-derived TTL" >&2
    exit 1
  fi
fi

echo "Results written to $output_dir"
