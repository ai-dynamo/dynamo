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
ENABLE_DETERMINISTIC_INFERENCE=${ENABLE_DETERMINISTIC_INFERENCE:-0}
HTTP_PORT=${HTTP_PORT:-18000}
STREAM_INTERVAL=${STREAM_INTERVAL:-1}
MIGRATE_AFTER_TOKENS=${MIGRATE_AFTER_TOKENS:-8}
DESTINATION_START_DELAY_MS=${DESTINATION_START_DELAY_MS:-500}
LOG_DIR=${LOG_DIR:-/tmp/decode-migration-logs}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

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
    wait 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM

# File discovery uses a 10-second TTL. First-request kernel warmup can exceed
# that while the worker is otherwise healthy, so keep local instance files
# fresh for the lifetime of this disposable container.
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

CUDA_VISIBLE_DEVICES="$SOURCE_GPUS" DYN_SYSTEM_PORT=18081 python3 -m dynamo.sglang \
    --endpoint dyn://dynamo.fast.generate \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-fast-worker \
    --decode-migration-class fast \
    --tp "$SOURCE_TP" \
    --page-size 16 \
    --host 0.0.0.0 \
    --port 18101 \
    --disaggregation-bootstrap-port 18201 \
    --disaggregation-transfer-backend nixl \
    --enable-decode-migration \
    --disable-overlap-schedule \
    --stream-interval "$STREAM_INTERVAL" \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    "${attention_args[@]}" \
    "${deterministic_args[@]}" \
    >"$LOG_DIR/fast.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES="$DESTINATION_GPUS" DYN_SYSTEM_PORT=18082 python3 -m dynamo.sglang \
    --endpoint dyn://dynamo.slow.generate \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-slow-worker \
    --decode-migration-class slow \
    --tp "$DESTINATION_TP" \
    --page-size 16 \
    --host 0.0.0.0 \
    --port 18102 \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port 18202 \
    --disaggregation-transfer-backend nixl \
    --enable-decode-migration \
    --disable-overlap-schedule \
    --stream-interval "$STREAM_INTERVAL" \
    --mem-fraction-static 0.5 \
    --disable-cuda-graph \
    "${attention_args[@]}" \
    "${deterministic_args[@]}" \
    >"$LOG_DIR/slow.log" 2>&1 &
pids+=("$!")

python3 -m dynamo.sglang.decode_migration_frontend \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --page-size 16 \
    --migrate-after-tokens "$MIGRATE_AFTER_TOKENS" \
    --destination-start-delay-ms "$DESTINATION_START_DELAY_MS" \
    >"$LOG_DIR/coordinator.log" 2>&1 &
pids+=("$!")

python3 -m dynamo.sglang.decode_migration_frontend \
    --namespace dynamo \
    --component baseline \
    --worker-namespace dynamo \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-baseline \
    --page-size 16 \
    --migrate-after-tokens 0 \
    >"$LOG_DIR/coordinator-baseline.log" 2>&1 &
pids+=("$!")

python3 -m dynamo.sglang.decode_migration_frontend \
    --namespace dynamo \
    --component rollback \
    --worker-namespace dynamo \
    --model-path "$MODEL_PATH" \
    --served-model-name decode-migration-rollback \
    --page-size 16 \
    --migrate-after-tokens "$MIGRATE_AFTER_TOKENS" \
    --destination-start-delay-ms "$DESTINATION_START_DELAY_MS" \
    --force-destination-failure \
    >"$LOG_DIR/coordinator-rollback.log" 2>&1 &
pids+=("$!")

python3 "$(dirname "$0")/test_scenarios.py" \
    --base-url "http://127.0.0.1:${HTTP_PORT}" \
    --model "$SERVED_MODEL_NAME" \
    --baseline-model decode-migration-baseline \
    --rollback-model decode-migration-rollback \
    --log-dir "$LOG_DIR" \
    --migrate-after-tokens "$MIGRATE_AFTER_TOKENS" \
    --stream-interval "$STREAM_INTERVAL"
