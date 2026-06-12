#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/qwen3-8b}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}
BASELINE_MODEL_NAME=${BASELINE_MODEL_NAME:-decode-migration-baseline}
SOURCE_TP=${SOURCE_TP:-4}
DESTINATION_TP=${DESTINATION_TP:-1}
SOURCE_GPUS=${SOURCE_GPUS:-0,1,2,3}
DESTINATION_GPUS=${DESTINATION_GPUS:-4}
HTTP_PORT=${HTTP_PORT:-28000}
STREAM_INTERVAL=${STREAM_INTERVAL:-1}
MIGRATE_ON_TOKEN_ID=${MIGRATE_ON_TOKEN_ID:-151668}
NUM_EXAMPLES=${NUM_EXAMPLES:-200}
NUM_THREADS=${NUM_THREADS:-1}
MAX_TOKENS=${MAX_TOKENS:-4096}
NUM_SHOTS=${NUM_SHOTS:-5}
MINIMUM_MIGRATION_COVERAGE=${MINIMUM_MIGRATION_COVERAGE:-1.0}
MINIMUM_REASONING_COVERAGE=${MINIMUM_REASONING_COVERAGE:-1.0}
MAX_SCORE_REGRESSION=${MAX_SCORE_REGRESSION:-0.02}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
RESULT_DIR=${RESULT_DIR:-/results}
LOG_DIR=${LOG_DIR:-${RESULT_DIR}/logs}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export PYTHONUNBUFFERED=1
export SGLANG_DISAGG_STAGING_BUFFER=0

attention_args=()
if [[ -n "$ATTENTION_BACKEND" ]]; then
    attention_args+=(--attention-backend "$ATTENTION_BACKEND")
fi

pids=()
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

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
    >"$LOG_DIR/frontend.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES="$SOURCE_GPUS" DYN_SYSTEM_PORT=28081 python3 -m dynamo.sglang \
    --endpoint dyn://dynamo.fast.generate \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-fast-worker \
    --decode-migration-class fast \
    --tp "$SOURCE_TP" \
    --page-size 16 \
    --host 0.0.0.0 \
    --port 28101 \
    --disaggregation-bootstrap-port 28201 \
    --disaggregation-transfer-backend nixl \
    --enable-decode-migration \
    --disable-overlap-schedule \
    --stream-interval "$STREAM_INTERVAL" \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-deterministic-inference \
    "${attention_args[@]}" \
    >"$LOG_DIR/fast.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES="$DESTINATION_GPUS" DYN_SYSTEM_PORT=28082 python3 -m dynamo.sglang \
    --endpoint dyn://dynamo.slow.generate \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-slow-worker \
    --decode-migration-class slow \
    --tp "$DESTINATION_TP" \
    --page-size 16 \
    --host 0.0.0.0 \
    --port 28102 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 28202 \
    --disaggregation-transfer-backend nixl \
    --enable-decode-migration \
    --disable-overlap-schedule \
    --stream-interval "$STREAM_INTERVAL" \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-deterministic-inference \
    "${attention_args[@]}" \
    >"$LOG_DIR/slow.log" 2>&1 &
pids+=("$!")

python3 -m dynamo.sglang.decode_migration_frontend \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --page-size 16 \
    --migrate-after-tokens 0 \
    --migrate-on-token-id "$MIGRATE_ON_TOKEN_ID" \
    --reasoning-parser qwen3 \
    --stream-interval "$STREAM_INTERVAL" \
    >"$LOG_DIR/coordinator.log" 2>&1 &
pids+=("$!")

python3 -m dynamo.sglang.decode_migration_frontend \
    --namespace dynamo \
    --component baseline \
    --worker-namespace dynamo \
    --model-path "$MODEL_PATH" \
    --served-model-name "$BASELINE_MODEL_NAME" \
    --page-size 16 \
    --migrate-after-tokens 0 \
    --reasoning-parser qwen3 \
    --stream-interval "$STREAM_INTERVAL" \
    >"$LOG_DIR/coordinator-baseline.log" 2>&1 &
pids+=("$!")

critical_pids=("${pids[@]:1}")
python3 /workspace/benchmark-driver/paired_qwen3_gsm8k.py \
    --base-url "http://127.0.0.1:${HTTP_PORT}" \
    --baseline-model "$BASELINE_MODEL_NAME" \
    --migrated-model "$SERVED_MODEL_NAME" \
    --num-examples "$NUM_EXAMPLES" \
    --num-threads "$NUM_THREADS" \
    --max-tokens "$MAX_TOKENS" \
    --num-shots "$NUM_SHOTS" \
    --migration-trigger-token-id "$MIGRATE_ON_TOKEN_ID" \
    --minimum-migration-coverage "$MINIMUM_MIGRATION_COVERAGE" \
    --minimum-reasoning-coverage "$MINIMUM_REASONING_COVERAGE" \
    --max-score-regression "$MAX_SCORE_REGRESSION" \
    --log-dir "$LOG_DIR" \
    --result-dir "$RESULT_DIR" &
eval_pid=$!
pids+=("$eval_pid")

while kill -0 "$eval_pid" 2>/dev/null; do
    for pid in "${critical_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Critical deployment process $pid exited before evaluation completed" >&2
            tail -80 "$LOG_DIR"/*.log >&2 || true
            exit 1
        fi
    done
    sleep 2
done
wait "$eval_pid"
