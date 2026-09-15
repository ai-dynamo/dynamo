#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one synchronized AIPerf side of the Qwen2.5 custom-encoder comparison.

set -euo pipefail

SIDE="${1:-}"
case "$SIDE" in
    control)
        ARM="custom-worker-control"
        TITLE="SEQUENTIAL CONTROL"
        DEFAULT_GPU_INDEX=0
        DEFAULT_URL=http://127.0.0.1:8000
        ;;
    dynamo-vllm)
        ARM="dynamo-vllm-custom"
        TITLE="DYNAMO.VLLM CUSTOM ENCODER"
        DEFAULT_GPU_INDEX=1
        DEFAULT_URL=http://127.0.0.1:8001
        ;;
    *)
        echo >&2 "Usage: $0 {control|dynamo-vllm}"
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COORDINATOR="$SCRIPT_DIR/demo_pair_coordinator.py"
WORKLOAD_ROOT="${DYN_DEMO_WORKLOAD_ROOT:-/dynamo-tmp/logs/08-05/qwen25-user-ensemble-textisl644-osl7-mixed1000/workload}"
MEASURED_INPUT="$WORKLOAD_ROOT/measured/image_custom_1000_textisl644.jsonl"
WARMUP_INPUT="$WORKLOAD_ROOT/warmup/image_custom_20_textisl644.jsonl"
MANIFEST="$WORKLOAD_ROOT/workload_manifest.json"
EXPECTED_INPUT_SHA256="743e859f895ee0e22df2476f74e5d3fa4d48db059273f5fe517634f31d9ef7cc"
URL="${DYN_DEMO_URL:-$DEFAULT_URL}"
GPU_INDEX="${DYN_DEMO_GPU_INDEX:-$DEFAULT_GPU_INDEX}"
CPUSET="${DYN_DEMO_CPUSET:-}"
CONCURRENCY="${DYN_DEMO_CONCURRENCY:-64}"
CONVERSATIONS="${DYN_DEMO_CONVERSATIONS:-1000}"
UI="${DYN_DEMO_UI:-simple}"
STATS_INTERVAL="${DYN_DEMO_STATS_INTERVAL:-5}"
RUN_ID="${DYN_DEMO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
DEMO_ROOT="${DYN_DEMO_OUTPUT_ROOT:-/dynamo-tmp/logs/$(date +%m-%d)/qwen25-custom-encoder-live-demo}"
OUTPUT_DIR="${DYN_DEMO_OUTPUT_DIR:-$DEMO_ROOT/$ARM-$RUN_ID}"
COORD_DIR="${DYN_DEMO_COORD_DIR:-$DEMO_ROOT/coord}"
COORD_SESSION="${DYN_DEMO_COORD_SESSION:-matched-h100-demo}"
SERIAL_SIDES="${DYN_DEMO_SERIAL_SIDES:-0}"
ZMQ_PREFIX="/tmp/aiperf-qwen25-demo-$ARM-$$"
TELEMETRY_PID=""
AIPERF_BIN="${DYN_DEMO_AIPERF_BIN:-}"

if [[ -z "$AIPERF_BIN" ]]; then
    if [[ -x /dynamo-tmp/venvs/aiperf-demo/bin/aiperf ]]; then
        AIPERF_BIN=/dynamo-tmp/venvs/aiperf-demo/bin/aiperf
    else
        AIPERF_BIN="$(command -v aiperf || true)"
    fi
fi

cleanup() {
    if [[ -n "$TELEMETRY_PID" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for command in curl jq nvidia-smi python sha256sum taskset; do
    command -v "$command" >/dev/null
done
if [[ -z "$AIPERF_BIN" || ! -x "$AIPERF_BIN" ]]; then
    echo >&2 "AIPerf executable not found; set DYN_DEMO_AIPERF_BIN"
    exit 1
fi
if ! "$AIPERF_BIN" --help >/dev/null 2>&1; then
    echo >&2 "AIPerf is not runnable: $AIPERF_BIN"
    exit 1
fi
for path in "$MEASURED_INPUT" "$WARMUP_INPUT" "$MANIFEST" "$COORDINATOR"; do
    test -f "$path"
done
test ! -e "$OUTPUT_DIR"

actual_sha256="$(sha256sum "$MEASURED_INPUT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_INPUT_SHA256" ]]; then
    echo >&2 "Workload SHA-256 mismatch: $actual_sha256"
    exit 1
fi
warmup_sha256="$(sha256sum "$WARMUP_INPUT" | awk '{print $1}')"
manifest_warmup_sha256="$(jq -er '.warmup.sha256' "$MANIFEST")"
if [[ "$warmup_sha256" != "$manifest_warmup_sha256" ]]; then
    echo >&2 "Warmup workload SHA-256 mismatch: $warmup_sha256"
    exit 1
fi

gpu_record="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,driver_version,uuid,pci.bus_id \
    --format=csv,noheader,nounits)"
IFS=',' read -r _ gpu_name gpu_memory gpu_power gpu_clock _ _ _ <<< "$gpu_record"
gpu_name="${gpu_name# }"
gpu_memory="${gpu_memory# }"
gpu_power="${gpu_power# }"
gpu_clock="${gpu_clock# }"
if [[ "$gpu_name" != "NVIDIA H100 80GB HBM3" \
    || "$gpu_memory" != "81559" \
    || "$gpu_power" != "700.00" \
    || "$gpu_clock" != "1980" ]]; then
    echo >&2 "Unexpected GPU hardware: $gpu_record"
    exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    python -c 'import torch; assert torch.cuda.device_count() == 1'

models_json="$(curl -fsS "$URL/v1/models")"
served_model="$(jq -er \
    '.data[0].id | select(type == "string" and length > 0)' \
    <<< "$models_json")"

mkdir -p "$OUTPUT_DIR/warmup" "$OUTPUT_DIR/measured" "$COORD_DIR"
printf '%s\n' "$gpu_record" > "$OUTPUT_DIR/gpu.txt"
printf '%s\n' "$actual_sha256" > "$OUTPUT_DIR/workload_sha256.txt"
printf '%s\n' "$served_model" > "$OUTPUT_DIR/served_model.txt"
printf '%s\n' "${CPUSET:-unrestricted}" > "$OUTPUT_DIR/cpuset.txt"
cp "$MANIFEST" "$OUTPUT_DIR/workload_manifest.json"

nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=timestamp,index,uuid,power.draw,power.limit,clocks.sm,clocks.max.sm,utilization.gpu,utilization.memory,memory.used,temperature.gpu \
    --format=csv,noheader,nounits \
    -lms 1000 > "$OUTPUT_DIR/gpu_telemetry.csv" &
TELEMETRY_PID=$!

COMMON_AIPERF_ARGS=(
    --model "$served_model"
    --url "$URL"
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
    --no-server-metrics
    --use-server-token-count
)

run_profile() {
    local input_file=$1
    local conversations=$2
    local concurrency=$3
    local artifact_dir=$4
    local zmq_path=$5
    local ui=$6
    local profile_command=(
        "$AIPERF_BIN" profile
        "${COMMON_AIPERF_ARGS[@]}"
        --input-file "$input_file"
        --concurrency "$concurrency"
        --conversation-num "$conversations"
        --ui "$ui"
        --artifact-dir "$artifact_dir"
        --zmq-ipc-path "$zmq_path"
    )
    if [[ -n "$CPUSET" ]]; then
        taskset -c "$CPUSET" "${profile_command[@]}"
    else
        "${profile_command[@]}"
    fi
}

printf '\033[1;36m============================================================\033[0m\n'
printf '\033[1;36m  %s\033[0m\n' "$TITLE"
printf '\033[1;36m============================================================\033[0m\n'
printf 'GPU:         %s\n' "$gpu_record"
printf 'Endpoint:    %s\n' "$URL"
printf 'Workload:    1000 unique images, shared 644-token prompt\n'
printf 'Generation:  exactly 7 output tokens\n'
printf 'Load:        closed-loop concurrency %s\n' "$CONCURRENCY"
printf 'Artifacts:   %s\n\n' "$OUTPUT_DIR"

pair_json=""
round_id=""
if [[ "$SERIAL_SIDES" == 1 ]]; then
    pair_json="$(python "$COORDINATOR" wait-turn \
        --state-dir "$COORD_DIR" \
        --session-id "$COORD_SESSION" \
        --side "$SIDE")"
    round_id="$(jq -er '.round_id' <<< "$pair_json")"
    printf '%s\n' "$round_id" > "$OUTPUT_DIR/pair_round.txt"
fi

printf 'Running 20 excluded warmup requests...\n'
export AIPERF_UI_REALTIME_METRICS_INTERVAL="$STATS_INTERVAL"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:pydantic.main}"
run_profile \
    "$WARMUP_INPUT" 20 20 \
    "$OUTPUT_DIR/warmup" "${ZMQ_PREFIX}-warmup" simple \
    > "$OUTPUT_DIR/warmup/client.log" 2>&1
jq -e \
    '(.request_count.avg == 20)
     and ((.error_summary | length) == 0)
     and (.was_cancelled == false)
     and (.output_sequence_length.avg == 7)' \
    "$OUTPUT_DIR/warmup/profile_export_aiperf.json" >/dev/null

if [[ "$SERIAL_SIDES" != 1 ]]; then
    pair_json="$(python "$COORDINATOR" wait-start \
        --state-dir "$COORD_DIR" \
        --session-id "$COORD_SESSION" \
        --side "$SIDE")"
    round_id="$(jq -er '.round_id' <<< "$pair_json")"
    printf '%s\n' "$round_id" > "$OUTPUT_DIR/pair_round.txt"
fi

printf '\nStarting measured AIPerf live view...\n\n'
start_ns="$(date +%s%N)"
run_profile \
    "$MEASURED_INPUT" "$CONVERSATIONS" "$CONCURRENCY" \
    "$OUTPUT_DIR/measured" "${ZMQ_PREFIX}-measured" "$UI"
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v start="$start_ns" -v end="$end_ns" \
    'BEGIN { printf "%.6f", (end - start) / 1000000000 }')"

result="$OUTPUT_DIR/measured/profile_export_aiperf.json"
jq -e \
    --argjson expected "$CONVERSATIONS" \
    '(.request_count.avg == $expected)
     and ((.error_summary | length) == 0)
     and (.was_cancelled == false)
     and (.output_sequence_length.avg == 7)' \
    "$result" >/dev/null

cleanup
TELEMETRY_PID=""
printf '%s\n' "$wall_seconds" > "$OUTPUT_DIR/measured/full_process_wall_seconds.txt"
full_process_throughput="$(awk \
    -v requests="$CONVERSATIONS" -v seconds="$wall_seconds" \
    'BEGIN { printf "%.6f", requests / seconds }')"

printf '\n\033[1;32m%s COMPLETE\033[0m\n' "$TITLE"
jq -r \
    --arg wall "$wall_seconds" \
    --arg full "$full_process_throughput" '
    "Requests:                 \(.request_count.avg | floor)",
    "Errors:                   \(.error_summary | length)",
    "Request throughput:       \(.request_throughput.avg) req/s",
    "Full-process throughput:  \($full) req/s",
    "Full-process wall time:   \($wall) s",
    "Output throughput:        \(.output_token_throughput.avg) tok/s",
    "Average E2E latency:      \(.request_latency.avg) ms",
    "P99 E2E latency:          \(.request_latency.p99) ms"
' "$result"

combined_json="$(python "$COORDINATOR" submit-result \
    --state-dir "$COORD_DIR" \
    --session-id "$COORD_SESSION" \
    --side "$SIDE" \
    --round-id "$round_id" \
    --result-path "$result" \
    --wall-seconds "$wall_seconds")"
printf '%s\n' "$combined_json" | jq . > "$OUTPUT_DIR/paired_comparison.json"

printf '\n\033[1;35mPAIRED H100 COMPARISON\033[0m\n'
jq -r '
    .comparison as $comparison |
    "Control throughput:        \(.control.request_throughput) req/s",
    "Dynamo throughput:         \(.dynamo_vllm.request_throughput) req/s",
    "Dynamo request gain:       \($comparison.request_throughput_gain_pct | if . >= 0 then "+" else "" end)\($comparison.request_throughput_gain_pct)%",
    "Dynamo full-wall gain:     \($comparison.full_process_throughput_gain_pct | if . >= 0 then "+" else "" end)\($comparison.full_process_throughput_gain_pct)%",
    "Average latency reduction: \($comparison.average_e2e_reduction_pct | if . >= 0 then "+" else "" end)\($comparison.average_e2e_reduction_pct)%",
    "P99 latency reduction:     \($comparison.p99_e2e_reduction_pct | if . >= 0 then "+" else "" end)\($comparison.p99_e2e_reduction_pct)%"
' <<< "$combined_json"
