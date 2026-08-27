#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the live AIPerf side of the Qwen2.5 custom-encoder comparison.

set -euo pipefail

SIDE="${1:-}"
case "$SIDE" in
    control)
        ARM="custom-worker-control"
        TITLE="SEQUENTIAL CONTROL"
        ;;
    dynamo-vllm)
        ARM="dynamo-vllm-custom"
        TITLE="DYNAMO.VLLM CUSTOM ENCODER"
        ;;
    *)
        echo >&2 "Usage: $0 {control|dynamo-vllm}"
        exit 2
        ;;
esac

WORKLOAD_ROOT="${DYN_DEMO_WORKLOAD_ROOT:-/dynamo-tmp/logs/08-05/qwen25-user-ensemble-textisl644-osl7-mixed1000/workload}"
INPUT_FILE="$WORKLOAD_ROOT/measured/image_custom_1000_textisl644.jsonl"
EXPECTED_INPUT_SHA256="743e859f895ee0e22df2476f74e5d3fa4d48db059273f5fe517634f31d9ef7cc"
URL="${DYN_DEMO_URL:-http://127.0.0.1:8000}"
CONCURRENCY="${DYN_DEMO_CONCURRENCY:-64}"
CONVERSATIONS="${DYN_DEMO_CONVERSATIONS:-1000}"
UI="${DYN_DEMO_UI:-dashboard}"
STATS_INTERVAL="${DYN_DEMO_STATS_INTERVAL:-5}"
RUN_ID="${DYN_DEMO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${DYN_DEMO_OUTPUT_DIR:-/dynamo-tmp/logs/$(date +%m-%d)/qwen25-custom-encoder-live-demo/$ARM-$RUN_ID}"
ZMQ_PREFIX="/tmp/aiperf-qwen25-demo-$ARM-$$"

for command in aiperf curl jq nvidia-smi sha256sum; do
    command -v "$command" >/dev/null
done
test -f "$INPUT_FILE"
test ! -e "$OUTPUT_DIR"

actual_sha256="$(sha256sum "$INPUT_FILE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_INPUT_SHA256" ]]; then
    echo >&2 "Workload SHA-256 mismatch: $actual_sha256"
    exit 1
fi

gpu_record="$(nvidia-smi \
    --query-gpu=name,memory.total,power.limit,clocks.max.sm,driver_version \
    --format=csv,noheader,nounits)"
IFS=',' read -r gpu_name gpu_memory gpu_power gpu_clock _ <<< "$gpu_record"
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
python -c 'import torch; assert torch.cuda.device_count() == 1'

models_json="$(curl -fsS "$URL/v1/models")"
served_model="$(jq -er \
    '.data[0].id | select(type == "string" and length > 0)' \
    <<< "$models_json")"

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$gpu_record" > "$OUTPUT_DIR/gpu.txt"
printf '%s\n' "$actual_sha256" > "$OUTPUT_DIR/workload_sha256.txt"
printf '%s\n' "$served_model" > "$OUTPUT_DIR/served_model.txt"

printf '\033[1;36m============================================================\033[0m\n'
printf '\033[1;36m  %s\033[0m\n' "$TITLE"
printf '\033[1;36m============================================================\033[0m\n'
printf 'GPU:         %s\n' "$gpu_record"
printf 'Workload:    1000 unique images, shared 644-token prompt\n'
printf 'Generation:  exactly 7 output tokens\n'
printf 'Load:        closed-loop concurrency %s\n' "$CONCURRENCY"
printf 'Artifacts:   %s\n\n' "$OUTPUT_DIR"

TIMEFORMAT='Full AIPerf process wall time: %3R seconds'
export AIPERF_UI_REALTIME_METRICS_INTERVAL="$STATS_INTERVAL"
time aiperf profile \
    --model "$served_model" \
    --url "$URL" \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --custom-dataset-type single_turn \
    --input-file "$INPUT_FILE" \
    --concurrency "$CONCURRENCY" \
    --conversation-num "$CONVERSATIONS" \
    --extra-inputs max_tokens:7 \
    --extra-inputs min_tokens:7 \
    --extra-inputs ignore_eos:true \
    --extra-inputs temperature:0 \
    --extra-inputs stream:false \
    --random-seed 42 \
    --workers-max 20 \
    --record-processors 32 \
    --request-timeout-seconds 300 \
    --ui "$UI" \
    --no-server-metrics \
    --use-server-token-count \
    --artifact-dir "$OUTPUT_DIR" \
    --zmq-ipc-path "$ZMQ_PREFIX"

result="$OUTPUT_DIR/profile_export_aiperf.json"
jq -e \
    --argjson expected "$CONVERSATIONS" \
    '(.request_count.avg == $expected)
     and ((.error_summary | length) == 0)
     and (.was_cancelled == false)
     and (.output_sequence_length.avg == 7)' \
    "$result" >/dev/null

printf '\n\033[1;32m%s COMPLETE\033[0m\n' "$TITLE"
jq -r '
    "Requests:             \(.request_count.avg | floor)",
    "Errors:               \(.error_summary | length)",
    "Request throughput:   \(.request_throughput.avg | tostring) req/s",
    "Output throughput:    \(.output_token_throughput.avg | tostring) tok/s",
    "Average E2E latency:  \(.request_latency.avg | tostring) ms",
    "P99 E2E latency:      \(.request_latency.p99 | tostring) ms"
' "$result"
