#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Qualify metadata-control and tensor-fanout classifier workflows at a constant
# offered load. Every repetition starts fresh server processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
LAUNCH_ROOT="${DYN_BENCH_LAUNCH_ROOT:-$REPO_ROOT}"
WORKLOAD_ROOT="${DYN_BENCH_WORKLOAD_ROOT:-/dynamo-tmp/logs/08-05/qwen25-user-ensemble-textisl644-osl7-mixed1000/workload}"
OUTPUT_ROOT="${DYN_BENCH_OUTPUT_ROOT:?set DYN_BENCH_OUTPUT_ROOT to a new result directory}"
CONTAINER_IMAGE="${DYN_BENCH_CONTAINER_IMAGE:?set DYN_BENCH_CONTAINER_IMAGE}"
AIPERF_BIN="${DYN_BENCH_AIPERF_BIN:-aiperf}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
ENCODER_SYSTEM_PORT="${DYN_ENCODER_SYSTEM_PORT:-8081}"
CLASSIFIER_SYSTEM_PORT="${DYN_CLASSIFIER_SYSTEM_PORT:-8082}"
DECODER_SYSTEM_PORT="${DYN_DECODER_SYSTEM_PORT:-8083}"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MEASURED_INPUT="$WORKLOAD_ROOT/measured/image_custom_1000_textisl644.jsonl"
WARMUP_INPUT="$WORKLOAD_ROOT/warmup/image_custom_20_textisl644.jsonl"
SERVER_PID=""
SAMPLER_PID=""
REQUEST_RATE="${DYN_BENCH_REQUEST_RATE:-50}"
KV_EVENT_PORT="${DYN_BENCH_KV_EVENT_PORT:-20080}"
KV_EVENTS_CONFIG="$(printf \
    '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%s","enable_kv_cache_events":true}' \
    "$KV_EVENT_PORT")"
DEFAULT_CELL_PLAN="1:metadata 2:metadata 3:metadata 1:tensor 2:tensor 3:tensor"
CELL_PLAN="${DYN_BENCH_CELL_PLAN:-$DEFAULT_CELL_PLAN}"
WORKFLOW_PROVIDER="${DYN_BENCH_WORKFLOW_PROVIDER:-examples.experimental.workflow.user_ensemble.benchmark.encoder_decoder_provider:provide_workflow}"

if [[ "$REQUEST_RATE" != 50 ]]; then
    echo >&2 "DYN_BENCH_REQUEST_RATE must be 50, got: $REQUEST_RATE"
    exit 2
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo >&2 "DYN_BENCH_OUTPUT_ROOT already exists: $OUTPUT_ROOT"
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_MODEL="$MODEL"
export DYN_SERVED_MODEL_NAME="$MODEL"
export DYN_ENCODER_MODEL="$MODEL"
export DYN_ENCODER_CLASS=examples.custom_encoder.qwen2_5_vl_benchmark_encoder.Qwen2_5VLBenchmarkEncoder
export DYN_QWEN2_VL_ENCODER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE=1536
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_PATCHES=41472
export DYN_QWEN2_VL_MAX_BATCH_ITEMS=64
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8,16,32,64
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=300x300,500x500
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0
export DYN_CUSTOM_ENCODER_DISPATCH_LOG=1
export DYN_ENCODER_BATCH_QUEUE_WAIT_MS="${DYN_ENCODER_BATCH_QUEUE_WAIT_MS:-0}"
export DYN_ENCODER_BATCH_QUEUE_MAX_WAIT_MS="${DYN_ENCODER_BATCH_QUEUE_MAX_WAIT_MS:-0}"
export DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-write
export DYN_CLASSIFIER_NIXL_BUFFER_BYTES="${DYN_CLASSIFIER_NIXL_BUFFER_BYTES:-536870912}"
export DYN_BENCH_SKIP_CLASSIFIER=0
export DYN_WORKFLOW_PERF_TRACE="${DYN_WORKFLOW_PERF_TRACE:-1}"
export DYN_WORKFLOW_PERF_SAMPLE_EVERY="${DYN_WORKFLOW_PERF_SAMPLE_EVERY:-32}"
if [[ "$DYN_WORKFLOW_PERF_TRACE" == 1 ]]; then
    export DYN_LOG="${DYN_LOG:-info}"
else
    export DYN_LOG="${DYN_LOG:-warn}"
fi
export DYN_MAX_MODEL_LEN=2048
export DYN_MAX_NUM_SEQS=64
export DYN_VLLM_GPU_MEMORY_UTILIZATION=0.4
export DYN_WORKER_GPU=0
export DYN_ENCODER_GPU=0
export DYN_DECODER_GPU=0
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/components/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup_cell() {
    if [[ -n "$SAMPLER_PID" ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=""
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_cell EXIT

python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
    validate-workload "$WORKLOAD_ROOT" \
    --output "$OUTPUT_ROOT/workload_audit.json"

SOURCE_COMMIT="${DYN_BENCH_SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SOURCE_BRANCH="${DYN_BENCH_SOURCE_BRANCH:-$(git -C "$REPO_ROOT" branch --show-current)}"
SOURCE_BRANCH="${SOURCE_BRANCH:-detached}"
WORKING_DIFF_SHA256="${DYN_BENCH_WORKING_DIFF_SHA256:-$(git -C "$REPO_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')}"
GPU_INFO="$(nvidia-smi \
    -i 0 \
    --query-gpu=name,power.limit,clocks.max.sm,memory.total \
    --format=csv,noheader,nounits | sed -n '1p')"
TORCH_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
AIPERF_VERSION="${DYN_BENCH_AIPERF_VERSION:-$("$AIPERF_BIN" --version 2>&1)}"

python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
    capture-metadata \
    --output "$OUTPUT_ROOT/benchmark_metadata.json" \
    --source-commit "$SOURCE_COMMIT" \
    --source-branch "$SOURCE_BRANCH" \
    --working-diff-sha256 "$WORKING_DIFF_SHA256" \
    --container-image "$CONTAINER_IMAGE" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --gpu-info "$GPU_INFO" \
    --torch-gpu-count "$TORCH_GPU_COUNT" \
    --request-rate "$REQUEST_RATE" \
    --aiperf-version "$AIPERF_VERSION" \
    --prefix-caching 1 \
    --kv-event-publishing 1 \
    --kv-event-port "$KV_EVENT_PORT" \
    --batch-queue-wait-ms "$DYN_ENCODER_BATCH_QUEUE_WAIT_MS" \
    --batch-queue-max-wait-ms "$DYN_ENCODER_BATCH_QUEUE_MAX_WAIT_MS" \
    --embedding-transfer-mode "$DYN_VLLM_EMBEDDING_TRANSFER_MODE" \
    --classifier-nixl-buffer-bytes "$DYN_CLASSIFIER_NIXL_BUFFER_BYTES" \
    --workflow-provider "$WORKFLOW_PROVIDER" \
    --perf-trace "$DYN_WORKFLOW_PERF_TRACE" \
    --perf-sample-every "$DYN_WORKFLOW_PERF_SAMPLE_EVERY"

sha256sum \
    "$REPO_ROOT/examples/experimental/workflow/user_ensemble/workflow.py" \
    "$REPO_ROOT/examples/experimental/workflow/user_ensemble/stages.py" \
    "$REPO_ROOT/examples/experimental/workflow/user_ensemble/benchmark/encoder_decoder_provider.py" \
    "$REPO_ROOT/examples/experimental/workflow/user_ensemble/remote/classifier_worker.py" \
    "$REPO_ROOT/examples/experimental/workflow/user_ensemble/remote/launch.sh" \
    "$REPO_ROOT/components/src/dynamo/common/multimodal/embedding_transfer.py" \
    "$REPO_ROOT/components/src/dynamo/vllm/multimodal_utils/external_encoder.py" \
    "$REPO_ROOT/components/src/dynamo/vllm/workflow/components/embedding_transfer.py" \
    "$REPO_ROOT/components/src/dynamo/vllm/multimodal_utils/custom_encoder/batcher.py" \
    "$REPO_ROOT/components/src/dynamo/experimental/workflow/nixl.py" \
    "$REPO_ROOT/components/src/dynamo/experimental/workflow/perf.py" \
    "$REPO_ROOT/components/src/dynamo/experimental/workflow/remote.py" \
    "$REPO_ROOT/components/src/dynamo/experimental/workflow/runtime.py" \
    "$REPO_ROOT/lib/bindings/python/src/dynamo/nixl_connect/__init__.py" \
    "$REPO_ROOT/examples/custom_encoder/qwen2_5_vl_benchmark_encoder.py" \
    > "$OUTPUT_ROOT/source_files_sha256.txt"
sha256sum "$MEASURED_INPUT" "$WARMUP_INPUT" \
    > "$OUTPUT_ROOT/workload_files_sha256.txt"

common_aiperf_args=(
    --model "$MODEL"
    --url "http://127.0.0.1:$HTTP_PORT"
    --endpoint-type chat
    --endpoint /v1/chat/completions
    --custom-dataset-type single_turn
    --extra-inputs max_tokens:7
    --extra-inputs min_tokens:7
    --extra-inputs ignore_eos:true
    --extra-inputs temperature:0
    --extra-inputs stream:false
    --random-seed 42
    --workers-max 20
    --record-processors 32
    --request-timeout-seconds 300
    --ui none
    --no-server-metrics
    --use-server-token-count
)

launch_topology() {
    local topology="$1"
    local output_dir="$2"
    local launch_script="$LAUNCH_ROOT/examples/experimental/workflow/user_ensemble/remote/launch.sh"
    local launch_args=(
        --max-num-seqs 64
        --enable-prefix-caching
        --kv-events-config "$KV_EVENTS_CONFIG"
    )
    local workflow_namespace="dynamo-qwen-$topology-$(date +%s%N)"

    if [[ "$topology" != metadata && "$topology" != tensor ]]; then
        echo >&2 "unknown topology: $topology"
        return 2
    fi

    export VLLM_CACHE_ROOT="/tmp/vllm-cache-user-ensemble-$topology"
    export XDG_CACHE_HOME="/tmp/xdg-cache-user-ensemble-$topology"
    rm -rf "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    printf '%s\n' "$workflow_namespace" > "$output_dir/workflow_namespace.txt"

    setsid env \
        DYN_NAMESPACE="$workflow_namespace" \
        DYN_USER_ENSEMBLE_NAMESPACE="$workflow_namespace" \
        DYN_BENCH_CLASSIFIER_INPUT="$topology" \
        DYN_BENCH_WORKFLOW_PROVIDER="$WORKFLOW_PROVIDER" \
        bash "$launch_script" "${launch_args[@]}" \
        > "$output_dir/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_control_plane() {
    local output_dir="$1"
    local response
    for _ in $(seq 1 1200); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            sed -n '1,240p' "$output_dir/server.log" >&2
            return 1
        fi
        if endpoint_generate_ready "$ENCODER_SYSTEM_PORT" \
            && classifier_ready \
            && endpoint_generate_ready "$DECODER_SYSTEM_PORT" \
            && response=$(curl -fsS \
            "http://127.0.0.1:$HTTP_PORT/v1/models" 2>/dev/null) \
            && python -c '
import json
import sys

model = sys.argv[1]
payload = json.load(sys.stdin)
raise SystemExit(
    0 if any(item.get("id") == model for item in payload.get("data", [])) else 1
)
' "$MODEL" <<< "$response"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

classifier_ready() {
    [[ "$DYN_BENCH_SKIP_CLASSIFIER" == 1 ]] \
        || endpoint_generate_ready "$CLASSIFIER_SYSTEM_PORT"
}

endpoint_generate_ready() {
    local port="$1"
    local response
    response=$(curl -sS --max-time 2 \
        "http://127.0.0.1:$port/health" 2>/dev/null) || return 1
    python -c '
import json
import sys

payload = json.load(sys.stdin)
raise SystemExit(
    0 if payload.get("endpoints", {}).get("generate") == "ready" else 1
)
' <<< "$response"
}

run_cell() {
    local repetition="$1"
    local topology="$2"
    local output_dir="$OUTPUT_ROOT/rep-$repetition/$topology"
    local zmq_prefix="/tmp/aiperf-qwen-workflow-r${repetition}-${topology}"

    mkdir -p "$output_dir/warmup" "$output_dir/measured"
    cleanup_cell
    rm -rf -- "${zmq_prefix}-warmup"* "${zmq_prefix}-measured"* || true
    launch_topology "$topology" "$output_dir"
    wait_control_plane "$output_dir"

    # The warmup is also the real-inference readiness gate. The measured run is
    # not started unless all 20 image-bearing requests complete successfully.
    "$AIPERF_BIN" profile "${common_aiperf_args[@]}" \
        --input-file "$WARMUP_INPUT" \
        --concurrency 20 \
        --conversation-num 20 \
        --artifact-dir "$output_dir/warmup" \
        --zmq-ipc-path "${zmq_prefix}-warmup" \
        > "$output_dir/warmup/client.log" 2>&1
    python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
        validate-profile \
        --profile "$output_dir/warmup/profile_export_aiperf.json" \
        --expected-requests 20 \
        --output "$output_dir/warmup/audit.json"

    local measured_log_start
    measured_log_start=$(wc -c < "$output_dir/server.log")

    nvidia-smi \
        -i 0 \
        --query-gpu=timestamp,utilization.gpu,memory.used \
        --format=csv,noheader,nounits \
        -l 1 > "$output_dir/gpu_telemetry.csv" &
    SAMPLER_PID=$!

    TIMEFORMAT='%R'
    { time "$AIPERF_BIN" profile "${common_aiperf_args[@]}" \
        --input-file "$MEASURED_INPUT" \
        --request-rate "$REQUEST_RATE" \
        --request-rate-mode constant \
        --conversation-num 1000 \
        --artifact-dir "$output_dir/measured" \
        --zmq-ipc-path "${zmq_prefix}-measured" \
        > "$output_dir/measured/client.log" 2>&1; } \
        2> "$output_dir/full_client_process_wall_seconds.txt"

    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""

    tail -c "+$((measured_log_start + 1))" "$output_dir/server.log" \
        > "$output_dir/measured/server.log"

    # Validate the exact 20+1,000 encoder calls before the remote joined-response
    # smoke adds one intentionally excluded request to the server log.
    python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
        validate-cell \
        --profile "$output_dir/measured/profile_export_aiperf.json" \
        --records "$output_dir/measured/profile_export.jsonl" \
        --wall-seconds "$output_dir/full_client_process_wall_seconds.txt" \
        --server-log "$output_dir/server.log" \
        --perf-log "$output_dir/measured/server.log" \
        --gpu-telemetry "$output_dir/gpu_telemetry.csv" \
        --output "$output_dir/cell_audit.json"

    python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
        smoke \
        --input "$WARMUP_INPUT" \
        --output "$output_dir/joined_smoke.json" \
        --endpoint "http://127.0.0.1:$HTTP_PORT/v1/chat/completions" \
        --model "$MODEL"
    cleanup_cell
}

for cell in $CELL_PLAN; do
    repetition="${cell%%:*}"
    topology="${cell#*:}"
    if [[ "$repetition" == "$cell" \
        || ! "$repetition" =~ ^[1-3]$ \
        || ( "$topology" != metadata && "$topology" != tensor ) ]]; then
        echo >&2 "invalid DYN_BENCH_CELL_PLAN entry: $cell"
        exit 2
    fi
    run_cell "$repetition" "$topology"
done

if [[ "$CELL_PLAN" == "$DEFAULT_CELL_PLAN" ]]; then
    python -m examples.experimental.workflow.user_ensemble.benchmark.remote_qwen_benchmark \
        summarize "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary.log"
else
    printf '%s\n' "$CELL_PLAN" > "$OUTPUT_ROOT/partial_cell_plan.txt"
fi
