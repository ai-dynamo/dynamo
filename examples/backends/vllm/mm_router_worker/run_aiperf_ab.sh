#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"

# -----------------------------------------------------------------------------
# Config (override via env)
# -----------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-dyn_ab_$(date +%H%M%S)}"
DATASET="${DATASET:-${DYNAMO_ROOT}/benchmarks/multimodal/jsonl/50req_1img_60word_datauri_prefixreuse_6imgpool.jsonl}"
HTTP_PORT="${HTTP_PORT:-8000}"
REQUEST_COUNT="${REQUEST_COUNT:-50}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-3}"
OSL="${OSL:-1}"
CONCURRENCIES="${CONCURRENCIES:-1 2}"

AIPERF_BIN="${AIPERF_BIN:-aiperf}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_MM_ROUTER="${RUN_MM_ROUTER:-1}"
CLEAN_BEFORE_START="${CLEAN_BEFORE_START:-1}"
STOP_WAIT_SECS="${STOP_WAIT_SECS:-5}"
RESTART_STACK_EACH_CONCURRENCY="${RESTART_STACK_EACH_CONCURRENCY:-1}"
READINESS_TIMEOUT_SECS="${READINESS_TIMEOUT_SECS:-300}"
PROBE_TIMEOUT_SECS="${PROBE_TIMEOUT_SECS:-120}"
CURL_CONNECT_TIMEOUT_SECS="${CURL_CONNECT_TIMEOUT_SECS:-2}"
CURL_MAX_TIME_SECS="${CURL_MAX_TIME_SECS:-8}"

RESULT_ROOT="${RESULT_ROOT:-${DYNAMO_ROOT}/logs/aiperf_compare}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULT_ROOT}/${TIMESTAMP}"

STACK_PID=""
STACK_PGID=""

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing command: $1" >&2
        exit 1
    fi
}

wait_frontend_models() {
    local timeout_s="${1:-300}"
    local deadline=$((SECONDS + timeout_s))
    local url="http://127.0.0.1:${HTTP_PORT}/v1/models"
    echo "Waiting for frontend at ${url} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS \
            --connect-timeout "${CURL_CONNECT_TIMEOUT_SECS}" \
            --max-time "${CURL_MAX_TIME_SECS}" \
            "${url}" >/dev/null 2>&1
        then
            echo "Frontend is ready"
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for frontend (${url})" >&2
    return 1
}

wait_model_registered() {
    local timeout_s="${1:-300}"
    local deadline=$((SECONDS + timeout_s))
    local url="http://127.0.0.1:${HTTP_PORT}/v1/models"
    local last_debug_ts=0
    echo "Waiting for model registration: ${MODEL}"
    while (( SECONDS < deadline )); do
        if [[ -n "${STACK_PID}" ]] && ! kill -0 "${STACK_PID}" 2>/dev/null; then
            echo "Stack process exited before model was registered (pid=${STACK_PID})." >&2
            return 1
        fi

        local resp=""
        resp="$(curl -fsS \
            --connect-timeout "${CURL_CONNECT_TIMEOUT_SECS}" \
            --max-time "${CURL_MAX_TIME_SECS}" \
            "${url}" 2>/dev/null || true)"
        if [[ -z "${resp}" ]]; then
            sleep 1
            continue
        fi

        local has_exact_model=""
        has_exact_model="$(printf '%s' "${resp}" | python -c 'import json,sys
target=sys.argv[1]
doc=json.load(sys.stdin)
ids=[m.get("id","") for m in doc.get("data",[]) if isinstance(m,dict)]
raise SystemExit(0 if target in ids else 1)
' "$MODEL" 2>/dev/null && echo "1" || true)"
        if [[ -n "${has_exact_model}" ]]; then
            echo "Model is registered on frontend"
            return 0
        fi

        if (( SECONDS - last_debug_ts >= 5 )); then
            local seen_ids=""
            seen_ids="$(printf '%s' "${resp}" | python -c 'import json,sys
doc=json.load(sys.stdin)
ids=[m.get("id","") for m in doc.get("data",[]) if isinstance(m,dict)]
print(", ".join(ids) if ids else "<none>")
' 2>/dev/null || echo "<json-parse-error>")"
            echo "Still waiting for '${MODEL}'. Current ids: ${seen_ids}"
            last_debug_ts="${SECONDS}"
        fi

        sleep 1
    done
    echo "Current /v1/models response:" >&2
    curl -fsS \
        --connect-timeout "${CURL_CONNECT_TIMEOUT_SECS}" \
        --max-time "${CURL_MAX_TIME_SECS}" \
        "${url}" >&2 || true
    echo "Timed out waiting for model '${MODEL}' in ${url}" >&2
    return 1
}

wait_request_path_ready() {
    local timeout_s="${1:-120}"
    local deadline=$((SECONDS + timeout_s))
    local url="http://127.0.0.1:${HTTP_PORT}/v1/chat/completions"
    local body
    body="$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0}' "${MODEL}")"

    echo "Probing chat path with model '${MODEL}' ..."
    while (( SECONDS < deadline )); do
        local resp=""
        resp="$(curl -sS \
            --connect-timeout "${CURL_CONNECT_TIMEOUT_SECS}" \
            --max-time "${CURL_MAX_TIME_SECS}" \
            -H 'Content-Type: application/json' \
            -d "${body}" \
            "${url}" 2>/dev/null || true)"

        if [[ -z "${resp}" ]]; then
            sleep 1
            continue
        fi

        if printf '%s' "${resp}" | python -c 'import json,sys
doc=json.load(sys.stdin)
ok=isinstance(doc.get("choices"), list) and len(doc["choices"]) > 0
raise SystemExit(0 if ok else 1)
' >/dev/null 2>&1
        then
            echo "Chat path is ready"
            return 0
        fi

        sleep 1
    done

    echo "Timed out probing chat path '${url}' with model '${MODEL}'" >&2
    return 1
}

wait_pid_exit() {
    local pid="$1"
    local timeout_s="${2:-15}"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

stop_stack() {
    if [[ -n "${STACK_PID}" ]]; then
        echo "Stopping stack pid=${STACK_PID} ..."
        if [[ -n "${STACK_PGID}" ]]; then
            # Send signal to the whole process group started by setsid.
            kill -INT -- "-${STACK_PGID}" 2>/dev/null || true
        else
            kill -INT "${STACK_PID}" 2>/dev/null || true
        fi

        if ! wait_pid_exit "${STACK_PID}" 15; then
            echo "Stack did not exit on SIGINT, sending SIGTERM ..."
            if [[ -n "${STACK_PGID}" ]]; then
                kill -TERM -- "-${STACK_PGID}" 2>/dev/null || true
            else
                kill -TERM "${STACK_PID}" 2>/dev/null || true
            fi
        fi
        if ! wait_pid_exit "${STACK_PID}" 10; then
            echo "Stack did not exit on SIGTERM, sending SIGKILL ..."
            if [[ -n "${STACK_PGID}" ]]; then
                kill -KILL -- "-${STACK_PGID}" 2>/dev/null || true
            else
                kill -KILL "${STACK_PID}" 2>/dev/null || true
            fi
        fi

        wait "${STACK_PID}" 2>/dev/null || true
        STACK_PID=""
        STACK_PGID=""
        echo "Waiting ${STOP_WAIT_SECS}s for process/ports to settle ..."
        sleep "${STOP_WAIT_SECS}"
    fi
}

cleanup_local_processes() {
    echo "Cleaning existing local Dynamo processes ..."
    pkill -f "examples.backends.vllm.mm_router_worker" 2>/dev/null || true
    pkill -f "python -m dynamo.frontend" 2>/dev/null || true
    pkill -f "python -m dynamo.vllm" 2>/dev/null || true
    pkill -f "launch_two_workers.sh" 2>/dev/null || true
    pkill -f "launch_two_workers_rr_baseline.sh" 2>/dev/null || true
    sleep 2
}

start_stack() {
    local mode="$1"  # baseline | mm
    local launch_script=""

    if [[ "${mode}" == "baseline" ]]; then
        launch_script="${DYNAMO_ROOT}/examples/backends/vllm/mm_router_worker/launch_two_workers_rr_baseline.sh"
    elif [[ "${mode}" == "mm" ]]; then
        launch_script="${DYNAMO_ROOT}/examples/backends/vllm/mm_router_worker/launch_two_workers.sh"
    else
        echo "Unknown mode: ${mode}" >&2
        exit 1
    fi

    local log_file="${RUN_DIR}/${mode}/server.log"
    mkdir -p "$(dirname "${log_file}")"

    echo "Starting ${mode} stack ..."
    # Start each stack in its own session/process-group so we can stop cleanly.
    setsid env MODEL="${MODEL}" NAMESPACE="${NAMESPACE}" HTTP_PORT="${HTTP_PORT}" "${launch_script}" >"${log_file}" 2>&1 &
    STACK_PID=$!
    STACK_PGID="${STACK_PID}"

    if ! wait_frontend_models "${READINESS_TIMEOUT_SECS}"; then
        echo "---- ${mode} server.log (tail) ----" >&2
        tail -n 120 "${log_file}" >&2 || true
        return 1
    fi
    if ! wait_model_registered "${READINESS_TIMEOUT_SECS}"; then
        echo "---- ${mode} server.log (tail) ----" >&2
        tail -n 120 "${log_file}" >&2 || true
        return 1
    fi
    if ! wait_request_path_ready "${PROBE_TIMEOUT_SECS}"; then
        echo "---- ${mode} server.log (tail) ----" >&2
        tail -n 120 "${log_file}" >&2 || true
        return 1
    fi
}

run_profile_once() {
    local mode="$1"
    local c="$2"
    local out_dir="${RUN_DIR}/${mode}/c${c}"
    mkdir -p "${out_dir}"

    echo
    echo "=== Profiling mode=${mode}, concurrency=${c}, model=${MODEL} ==="
    "${AIPERF_BIN}" profile \
        --model "${MODEL}" \
        --input-file "${DATASET}" \
        --custom-dataset-type single_turn \
        --osl "${OSL}" \
        --request-count "${REQUEST_COUNT}" \
        --concurrency "${c}" \
        --warmup-request-count "${WARMUP_REQUEST_COUNT}" \
        --artifact-dir "${out_dir}" \
        --ui-type none
}

run_mode() {
    local mode="$1"
    if [[ "${CLEAN_BEFORE_START}" == "1" ]]; then
        cleanup_local_processes
    fi

    if [[ "${RESTART_STACK_EACH_CONCURRENCY}" == "1" ]]; then
        for c in ${CONCURRENCIES}; do
            start_stack "${mode}"
            run_profile_once "${mode}" "${c}"
            stop_stack
            if [[ "${CLEAN_BEFORE_START}" == "1" ]]; then
                cleanup_local_processes
            fi
        done
    else
        start_stack "${mode}"
        for c in ${CONCURRENCIES}; do
            run_profile_once "${mode}" "${c}"
        done
        stop_stack
    fi
}

trap 'stop_stack' EXIT INT TERM

require_cmd "${AIPERF_BIN}"
require_cmd curl

if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found: ${DATASET}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Model: ${MODEL}"
echo "Namespace: ${NAMESPACE}"
echo "Dataset: ${DATASET}"
echo "Concurrencies: ${CONCURRENCIES}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_mode baseline
fi

if [[ "${RUN_MM_ROUTER}" == "1" ]]; then
    run_mode mm
fi

echo
echo "Done. Results:"
echo "  ${RUN_DIR}"
echo
echo "Quick compare files:"
echo "  ${RUN_DIR}/baseline/c1/profile_export_aiperf.json"
echo "  ${RUN_DIR}/baseline/c2/profile_export_aiperf.json"
echo "  ${RUN_DIR}/mm/c1/profile_export_aiperf.json"
echo "  ${RUN_DIR}/mm/c2/profile_export_aiperf.json"
