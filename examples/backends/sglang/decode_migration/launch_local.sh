#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/qwen3-0.6b}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen/Qwen3-0.6B}
SOURCE_TP=${SOURCE_TP:-1}
DESTINATION_TP=${DESTINATION_TP:-1}
SOURCE_DP=${SOURCE_DP:-1}
DESTINATION_DP=${DESTINATION_DP:-1}
SOURCE_EP=${SOURCE_EP:-1}
DESTINATION_EP=${DESTINATION_EP:-1}
SOURCE_GPUS=${SOURCE_GPUS:-0}
DESTINATION_GPUS=${DESTINATION_GPUS:-1}
SOURCE_ENABLE_DP_ATTENTION=${SOURCE_ENABLE_DP_ATTENTION:-0}
DESTINATION_ENABLE_DP_ATTENTION=${DESTINATION_ENABLE_DP_ATTENTION:-0}
SOURCE_MOE_A2A_BACKEND=${SOURCE_MOE_A2A_BACKEND:-none}
DESTINATION_MOE_A2A_BACKEND=${DESTINATION_MOE_A2A_BACKEND:-none}
SOURCE_MOE_RUNNER_BACKEND=${SOURCE_MOE_RUNNER_BACKEND:-}
DESTINATION_MOE_RUNNER_BACKEND=${DESTINATION_MOE_RUNNER_BACKEND:-}
SOURCE_DEEPEP_MODE=${SOURCE_DEEPEP_MODE:-}
DESTINATION_DEEPEP_MODE=${DESTINATION_DEEPEP_MODE:-}
SOURCE_CUDA_GRAPH_BS=${SOURCE_CUDA_GRAPH_BS:-}
DESTINATION_CUDA_GRAPH_BS=${DESTINATION_CUDA_GRAPH_BS:-}
SOURCE_MAX_RUNNING_REQUESTS=${SOURCE_MAX_RUNNING_REQUESTS:-2048}
DESTINATION_MAX_RUNNING_REQUESTS=${DESTINATION_MAX_RUNNING_REQUESTS:-2048}
SOURCE_MEM_FRACTION_STATIC=${SOURCE_MEM_FRACTION_STATIC:-0.5}
DESTINATION_MEM_FRACTION_STATIC=${DESTINATION_MEM_FRACTION_STATIC:-0.5}
SOURCE_ENABLE_JIT_DEEPGEMM=${SOURCE_ENABLE_JIT_DEEPGEMM:-0}
DESTINATION_ENABLE_JIT_DEEPGEMM=${DESTINATION_ENABLE_JIT_DEEPGEMM:-0}
DEEPEP_MAX_DISPATCH_TOKENS=${DEEPEP_MAX_DISPATCH_TOKENS:-1024}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}
ENABLE_DETERMINISTIC_INFERENCE=${ENABLE_DETERMINISTIC_INFERENCE:-1}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
HTTP_PORT=${HTTP_PORT:-18000}
SOURCE_SYSTEM_PORT=${SOURCE_SYSTEM_PORT:-18081}
DESTINATION_SYSTEM_PORT=${DESTINATION_SYSTEM_PORT:-18082}
SOURCE_PORT=${SOURCE_PORT:-18101}
DESTINATION_PORT=${DESTINATION_PORT:-18102}
SOURCE_BOOTSTRAP_PORT=${SOURCE_BOOTSTRAP_PORT:-18201}
DESTINATION_BOOTSTRAP_PORT=${DESTINATION_BOOTSTRAP_PORT:-18202}
SOURCE_DIST_INIT_ADDR=${SOURCE_DIST_INIT_ADDR:-}
DESTINATION_DIST_INIT_ADDR=${DESTINATION_DIST_INIT_ADDR:-}
STREAM_INTERVAL=${STREAM_INTERVAL:-1}
PARITY_MODE=${PARITY_MODE:-}
WORKER_LOG_LEVEL=${WORKER_LOG_LEVEL:-info}
MIGRATE_AFTER_TOKENS=${MIGRATE_AFTER_TOKENS:-8}
TEST_MODE=${TEST_MODE:-correctness}
GSM8K_NUM_QUESTIONS=${GSM8K_NUM_QUESTIONS:-20}
GSM8K_INDICES=${GSM8K_INDICES:-}
GSM8K_MAX_ATTEMPTS=${GSM8K_MAX_ATTEMPTS:-}
GSM8K_NUM_SHOTS=${GSM8K_NUM_SHOTS:-5}
GSM8K_MAX_TOKENS=${GSM8K_MAX_TOKENS:-1024}
GSM8K_ALLOWED_REGRESSIONS=${GSM8K_ALLOWED_REGRESSIONS:-1}
GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-}
LOG_DIR=${LOG_DIR:-/tmp/decode-migration-logs}

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

if [[ -z "$PARITY_MODE" ]]; then
    if [[ "$SOURCE_TP" == "$DESTINATION_TP" \
        && "$SOURCE_DP" == "$DESTINATION_DP" \
        && "$SOURCE_EP" == "$DESTINATION_EP" \
        && "$SOURCE_ENABLE_DP_ATTENTION" == "$DESTINATION_ENABLE_DP_ATTENTION" ]]; then
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
cuda_graph_args=()
if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    cuda_graph_args+=(--disable-cuda-graph)
fi

source_topology_args=(
    --tensor-parallel-size "$SOURCE_TP"
    --data-parallel-size "$SOURCE_DP"
    --expert-parallel-size "$SOURCE_EP"
    --moe-a2a-backend "$SOURCE_MOE_A2A_BACKEND"
    --max-running-requests "$SOURCE_MAX_RUNNING_REQUESTS"
    --mem-fraction-static "$SOURCE_MEM_FRACTION_STATIC"
)
destination_topology_args=(
    --tensor-parallel-size "$DESTINATION_TP"
    --data-parallel-size "$DESTINATION_DP"
    --expert-parallel-size "$DESTINATION_EP"
    --moe-a2a-backend "$DESTINATION_MOE_A2A_BACKEND"
    --max-running-requests "$DESTINATION_MAX_RUNNING_REQUESTS"
    --mem-fraction-static "$DESTINATION_MEM_FRACTION_STATIC"
)
if [[ "$SOURCE_ENABLE_DP_ATTENTION" == "1" ]]; then
    source_topology_args+=(
        --enable-dp-attention
        --enable-dp-attention-local-control-broadcast
    )
fi
if [[ "$DESTINATION_ENABLE_DP_ATTENTION" == "1" ]]; then
    destination_topology_args+=(
        --enable-dp-attention
        --enable-dp-attention-local-control-broadcast
    )
fi
if [[ -n "$SOURCE_DIST_INIT_ADDR" ]]; then
    source_topology_args+=(--dist-init-addr "$SOURCE_DIST_INIT_ADDR")
fi
if [[ -n "$DESTINATION_DIST_INIT_ADDR" ]]; then
    destination_topology_args+=(--dist-init-addr "$DESTINATION_DIST_INIT_ADDR")
fi
if [[ -n "$SOURCE_MOE_RUNNER_BACKEND" ]]; then
    source_topology_args+=(--moe-runner-backend "$SOURCE_MOE_RUNNER_BACKEND")
fi
if [[ -n "$DESTINATION_MOE_RUNNER_BACKEND" ]]; then
    destination_topology_args+=(--moe-runner-backend "$DESTINATION_MOE_RUNNER_BACKEND")
fi
if [[ -n "$SOURCE_DEEPEP_MODE" ]]; then
    source_topology_args+=(--deepep-mode "$SOURCE_DEEPEP_MODE")
fi
if [[ -n "$DESTINATION_DEEPEP_MODE" ]]; then
    destination_topology_args+=(--deepep-mode "$DESTINATION_DEEPEP_MODE")
fi
if [[ -n "$SOURCE_CUDA_GRAPH_BS" ]]; then
    read -r -a source_cuda_graph_bs <<<"$SOURCE_CUDA_GRAPH_BS"
    source_topology_args+=(--cuda-graph-bs-decode "${source_cuda_graph_bs[@]}")
fi
if [[ -n "$DESTINATION_CUDA_GRAPH_BS" ]]; then
    read -r -a destination_cuda_graph_bs <<<"$DESTINATION_CUDA_GRAPH_BS"
    destination_topology_args+=(--cuda-graph-bs-decode "${destination_cuda_graph_bs[@]}")
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

wait_for_log_count() {
    local log_file=$1
    local text=$2
    local minimum=$3
    local timeout=${4:-180}
    local deadline=$((SECONDS + timeout))

    while (( SECONDS < deadline )); do
        if (( $(grep -F -c "$text" "$log_file" 2>/dev/null || true) >= minimum )); then
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for $minimum occurrence(s) of '$text' in $log_file" >&2
    tail -80 "$log_file" >&2 || true
    return 1
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
    --stream-interval "$STREAM_INTERVAL"
    --log-level "$WORKER_LOG_LEVEL"
    "${cuda_graph_args[@]}"
)

CUDA_VISIBLE_DEVICES="$SOURCE_GPUS" \
SGLANG_ENABLE_JIT_DEEPGEMM="$SOURCE_ENABLE_JIT_DEEPGEMM" \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$DEEPEP_MAX_DISPATCH_TOKENS" \
DYN_SYSTEM_PORT="$SOURCE_SYSTEM_PORT" python3 -m dynamo.sglang \
    "${common_worker_args[@]}" \
    --worker-taint decode/fast \
    "${source_topology_args[@]}" \
    --port "$SOURCE_PORT" \
    --disaggregation-bootstrap-port "$SOURCE_BOOTSTRAP_PORT" \
    "${attention_args[@]}" \
    "${deterministic_args[@]}" \
    >"$LOG_DIR/fast.log" 2>&1 &
source_pid=$!
pids+=("$source_pid")

CUDA_VISIBLE_DEVICES="$DESTINATION_GPUS" \
SGLANG_ENABLE_JIT_DEEPGEMM="$DESTINATION_ENABLE_JIT_DEEPGEMM" \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$DEEPEP_MAX_DISPATCH_TOKENS" \
DYN_SYSTEM_PORT="$DESTINATION_SYSTEM_PORT" python3 -m dynamo.sglang \
    "${common_worker_args[@]}" \
    --worker-taint decode/slow \
    "${destination_topology_args[@]}" \
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
wait_for_log_count "$LOG_DIR/fast.log" "Model registration succeeded" 1 900
wait_for_log_count "$LOG_DIR/slow.log" "Model registration succeeded" 1 900
wait_for_log_count "$LOG_DIR/frontend.log" "Adding worker WorkerWithDpRank" 2

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
        if [[ -n "$GSM8K_INDICES" ]]; then
            gsm8k_args+=(--indices "$GSM8K_INDICES")
        fi
        if [[ -n "$GSM8K_DATA_PATH" ]]; then
            gsm8k_args+=(--data-path "$GSM8K_DATA_PATH")
        fi
        python3 "$(dirname "$0")/qwen3_gsm8k_accuracy.py" "${gsm8k_args[@]}"
        ;;
    serve)
        echo "Dynamo frontend and migration workers are ready on port $HTTP_PORT"
        echo "Press Ctrl-C to stop the deployment"
        wait "$frontend_pid"
        ;;
    *)
        echo "Unknown TEST_MODE=$TEST_MODE; expected correctness, gsm8k, or serve" >&2
        exit 2
        ;;
esac
