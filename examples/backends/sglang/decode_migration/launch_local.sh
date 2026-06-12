#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/qwen3-0.6b}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-0.6B}
SOURCE_TP=${SOURCE_TP:-1}
DESTINATION_TP=${DESTINATION_TP:-1}
SOURCE_GPUS=${SOURCE_GPUS:-0}
DESTINATION_GPUS=${DESTINATION_GPUS:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}
ENABLE_DETERMINISTIC_INFERENCE=${ENABLE_DETERMINISTIC_INFERENCE:-1}
HTTP_PORT=${HTTP_PORT:-18000}
SOURCE_SYSTEM_PORT=${SOURCE_SYSTEM_PORT:-18081}
DESTINATION_SYSTEM_PORT=${DESTINATION_SYSTEM_PORT:-18082}
SOURCE_PORT=${SOURCE_PORT:-18101}
DESTINATION_PORT=${DESTINATION_PORT:-18102}
SOURCE_BOOTSTRAP_PORT=${SOURCE_BOOTSTRAP_PORT:-18201}
DESTINATION_BOOTSTRAP_PORT=${DESTINATION_BOOTSTRAP_PORT:-18202}
STREAM_INTERVAL=${STREAM_INTERVAL:-1}
PARITY_MODE=${PARITY_MODE:-}
WORKER_LOG_LEVEL=${WORKER_LOG_LEVEL:-info}
MIGRATE_AFTER_TOKENS=${MIGRATE_AFTER_TOKENS:-8}
TEST_MODE=${TEST_MODE:-correctness}
GSM8K_NUM_QUESTIONS=${GSM8K_NUM_QUESTIONS:-20}
GSM8K_MAX_ATTEMPTS=${GSM8K_MAX_ATTEMPTS:-}
GSM8K_NUM_SHOTS=${GSM8K_NUM_SHOTS:-5}
GSM8K_MAX_TOKENS=${GSM8K_MAX_TOKENS:-1024}
GSM8K_ALLOWED_REGRESSIONS=${GSM8K_ALLOWED_REGRESSIONS:-1}
GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-}
LOG_DIR=${LOG_DIR:-/tmp/decode-migration-logs}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

if [[ -z "$PARITY_MODE" ]]; then
    if [[ "$SOURCE_TP" == "$DESTINATION_TP" ]]; then
        PARITY_MODE=source
    else
        PARITY_MODE=migration-repeat
    fi
fi

for system_port in "$SOURCE_SYSTEM_PORT" "$DESTINATION_SYSTEM_PORT"; do
    if (( system_port < 1 || system_port > 32767 )); then
        echo "Dynamo system ports must be in [1, 32767], got $system_port" >&2
        exit 2
    fi
done

python3 "$(dirname "$0")/check_ports.py" \
    "$HTTP_PORT" \
    "$SOURCE_SYSTEM_PORT" "$DESTINATION_SYSTEM_PORT" \
    "$SOURCE_PORT" "$DESTINATION_PORT" \
    "$SOURCE_BOOTSTRAP_PORT" "$DESTINATION_BOOTSTRAP_PORT"

export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-file}
export DYN_REQUEST_PLANE=${DYN_REQUEST_PLANE:-tcp}
export PYTHONUNBUFFERED=1
export SGLANG_DISAGG_STAGING_BUFFER=${SGLANG_DISAGG_STAGING_BUFFER:-0}

pids=()
attention_args=()
if [[ -n "$ATTENTION_BACKEND" ]]; then
    attention_args+=(--attention-backend "$ATTENTION_BACKEND")
fi
deterministic_args=()
if [[ "$ENABLE_DETERMINISTIC_INFERENCE" == "1" ]]; then
    deterministic_args+=(--enable-deterministic-inference)
fi
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${pids[@]:-}"; do
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    exit "$rc"
}
trap cleanup EXIT INT TERM

check_started() {
    local pid=$1
    local name=$2
    local log_file=$3
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name exited during startup" >&2
        tail -80 "$log_file" >&2 || true
        return 1
    fi
}

# File discovery uses a short TTL. Keep local registrations fresh during model warmup.
(
    while sleep 1; do
        find /tmp/dynamo_store_kv/v1/instances -type f -exec touch {} + \
            2>/dev/null || true
    done
) &
pids+=("$!")

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --discovery-backend file \
    --namespace dynamo \
    --router-mode kv \
    --enable-decode-migration \
    >"$LOG_DIR/frontend.log" 2>&1 &
frontend_pid=$!
pids+=("$frontend_pid")

common_worker_args=(
    --endpoint dyn://dynamo.backend.generate
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --page-size 16
    --host 0.0.0.0
    --disaggregation-transfer-backend nixl
    --enable-decode-migration
    --disable-overlap-schedule
    --stream-interval "$STREAM_INTERVAL"
    --log-level "$WORKER_LOG_LEVEL"
    --mem-fraction-static 0.5
    --disable-cuda-graph
)

CUDA_VISIBLE_DEVICES="$SOURCE_GPUS" DYN_SYSTEM_PORT="$SOURCE_SYSTEM_PORT" python3 -m dynamo.sglang \
    "${common_worker_args[@]}" \
    --worker-taint decode/fast \
    --tp "$SOURCE_TP" \
    --port "$SOURCE_PORT" \
    --disaggregation-bootstrap-port "$SOURCE_BOOTSTRAP_PORT" \
    "${attention_args[@]}" \
    "${deterministic_args[@]}" \
    >"$LOG_DIR/fast.log" 2>&1 &
source_pid=$!
pids+=("$source_pid")

CUDA_VISIBLE_DEVICES="$DESTINATION_GPUS" DYN_SYSTEM_PORT="$DESTINATION_SYSTEM_PORT" python3 -m dynamo.sglang \
    "${common_worker_args[@]}" \
    --worker-taint decode/slow \
    --tp "$DESTINATION_TP" \
    --port "$DESTINATION_PORT" \
    --disaggregation-bootstrap-port "$DESTINATION_BOOTSTRAP_PORT" \
    "${attention_args[@]}" \
    "${deterministic_args[@]}" \
    >"$LOG_DIR/slow.log" 2>&1 &
destination_pid=$!
pids+=("$destination_pid")

sleep 2
check_started "$frontend_pid" frontend "$LOG_DIR/frontend.log"
check_started "$source_pid" source-worker "$LOG_DIR/fast.log"
check_started "$destination_pid" destination-worker "$LOG_DIR/slow.log"

case "$TEST_MODE" in
    correctness)
        python3 "$(dirname "$0")/test_scenarios.py" \
            --base-url "http://127.0.0.1:${HTTP_PORT}" \
            --model "$SERVED_MODEL_NAME" \
            --tokenizer-path "$MODEL_PATH" \
            --log-dir "$LOG_DIR" \
            --migrate-after-tokens "$MIGRATE_AFTER_TOKENS" \
            --stream-interval "$STREAM_INTERVAL" \
            --parity-mode "$PARITY_MODE"
        ;;
    gsm8k)
        gsm8k_args=(
            --base-url "http://127.0.0.1:${HTTP_PORT}"
            --model "$SERVED_MODEL_NAME"
            --tokenizer-path "$MODEL_PATH"
            --log-dir "$LOG_DIR"
            --num-questions "$GSM8K_NUM_QUESTIONS"
            --num-shots "$GSM8K_NUM_SHOTS"
            --max-tokens "$GSM8K_MAX_TOKENS"
            --allowed-regressions "$GSM8K_ALLOWED_REGRESSIONS"
        )
        if [[ -n "$GSM8K_MAX_ATTEMPTS" ]]; then
            gsm8k_args+=(--max-attempts "$GSM8K_MAX_ATTEMPTS")
        fi
        if [[ -n "$GSM8K_DATA_PATH" ]]; then
            gsm8k_args+=(--data-path "$GSM8K_DATA_PATH")
        fi
        python3 "$(dirname "$0")/qwen3_gsm8k_accuracy.py" "${gsm8k_args[@]}"
        ;;
    *)
        echo "Unknown TEST_MODE=$TEST_MODE; expected correctness or gsm8k" >&2
        exit 2
        ;;
esac
