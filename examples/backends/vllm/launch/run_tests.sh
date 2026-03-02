#!/usr/bin/env bash
# run_tests.sh — QA harness for vLLM launch scripts
# Usage:
#   bash run_tests.sh                        # run all tests
#   bash run_tests.sh --only agg.sh          # run one script
#   bash run_tests.sh --start-from disagg.sh # resume from a given script
#
# Requires: curl, python3, nvidia-smi, setsid

set -uo pipefail
# NOTE: no set -e — a failing test must not abort the whole run

# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------
STARTUP_TIMEOUT=500        # seconds to wait for /v1/models
TOTAL_TEST_TIMEOUT=600     # hard wall-clock limit per test (seconds)
HEALTH_POLL_INTERVAL=5     # seconds between /v1/models polls
N_REQUESTS=3               # number of test requests per script
GPU_DRAIN_TIMEOUT=60       # max seconds to wait for GPU to clear
PORT_DRAIN_TIMEOUT=30      # max seconds to wait for ports to free
BETWEEN_TEST_SLEEP=10      # seconds between tests

BASE_PORT=8000             # primary OpenAI-compatible port
REPLICA_PORT=8001          # second port used by agg_router_replicas.sh
SYSTEM_PORTS=(8081 8082 8083 8091 8092)
ALL_PORTS=(8000 8001 8081 8082 8083 8091 8092)

MULTIMODAL_IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

# ---------------------------------------------------------------------------
# Script metadata
# ---------------------------------------------------------------------------

# Execution order (1-GPU → 2-GPU → 4-GPU → large models)
TEST_SCRIPTS=(
    agg.sh
    agg_kvbm.sh
    agg_lmcache.sh
    agg_lmcache_multiproc.sh
    agg_request_planes.sh
    disagg_same_gpu.sh
    disagg.sh
    disagg_kvbm.sh
    disagg_lmcache.sh
    agg_router.sh
    agg_router_approx.sh
    agg_router_replicas.sh
    disagg_router.sh
    disagg_kvbm_2p2d.sh
    disagg_kvbm_router.sh
    dep.sh
    agg_spec_decoding.sh
    agg_omni.sh
    agg_omni_image.sh
    agg_omni_video.sh
    agg_multimodal.sh
    disagg_multimodal_e_pd.sh
    disagg_multimodal_epd.sh
    vllm_serve_embedding_cache.sh
)

SKIP_SCRIPTS=(dsr1_dep.sh disagg_router_gaudi.sh multi_node_tp.sh disagg_multimodal_llama.sh)
EXPECTED_FAILURES=(dep.sh)

# Request shape: text | multimodal | omni_image | omni_video
declare -A SCRIPT_TYPE=(
    [agg.sh]=text
    [agg_kvbm.sh]=text
    [agg_lmcache.sh]=text
    [agg_lmcache_multiproc.sh]=text
    [agg_request_planes.sh]=text
    [disagg_same_gpu.sh]=text
    [disagg.sh]=text
    [disagg_kvbm.sh]=text
    [disagg_lmcache.sh]=text
    [agg_router.sh]=text
    [agg_router_approx.sh]=text
    [agg_router_replicas.sh]=text
    [disagg_router.sh]=text
    [disagg_kvbm_2p2d.sh]=text
    [disagg_kvbm_router.sh]=text
    [dep.sh]=text
    [agg_spec_decoding.sh]=text
    [agg_omni.sh]=text
    [agg_omni_image.sh]=omni_image
    [agg_omni_video.sh]=omni_video
    [agg_multimodal.sh]=multimodal
    [disagg_multimodal_e_pd.sh]=multimodal
    [disagg_multimodal_epd.sh]=multimodal
    [vllm_serve_embedding_cache.sh]=multimodal
)

# Model name passed in the request body
declare -A SCRIPT_MODEL=(
    [agg.sh]="Qwen/Qwen3-0.6B"
    [agg_kvbm.sh]="Qwen/Qwen3-0.6B"
    [agg_lmcache.sh]="Qwen/Qwen3-0.6B"
    [agg_lmcache_multiproc.sh]="Qwen/Qwen3-0.6B"
    [agg_request_planes.sh]="Qwen/Qwen3-0.6B"
    [disagg_same_gpu.sh]="Qwen/Qwen3-0.6B"
    [disagg.sh]="Qwen/Qwen3-0.6B"
    [disagg_kvbm.sh]="Qwen/Qwen3-0.6B"
    [disagg_lmcache.sh]="Qwen/Qwen3-0.6B"
    [agg_router.sh]="Qwen/Qwen3-0.6B"
    [agg_router_approx.sh]="Qwen/Qwen3-0.6B"
    [agg_router_replicas.sh]="Qwen/Qwen3-0.6B"
    [disagg_router.sh]="Qwen/Qwen3-0.6B"
    [disagg_kvbm_2p2d.sh]="Qwen/Qwen3-0.6B"
    [disagg_kvbm_router.sh]="Qwen/Qwen3-0.6B"
    [dep.sh]="Qwen/Qwen3-30B-A3B"
    [agg_spec_decoding.sh]="meta-llama/Meta-Llama-3.1-8B-Instruct"
    [agg_omni.sh]="Qwen/Qwen2.5-Omni-7B"
    [agg_omni_image.sh]="Qwen/Qwen-Image"
    [agg_omni_video.sh]="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    [agg_multimodal.sh]="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    [disagg_multimodal_e_pd.sh]="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    [disagg_multimodal_epd.sh]="llava-hf/llava-1.5-7b-hf"
    [vllm_serve_embedding_cache.sh]="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
)

# Extra CLI args to pass when launching a script
declare -A SCRIPT_EXTRA_ARGS=(
    [disagg_multimodal_e_pd.sh]="--single-gpu"
    [disagg_multimodal_epd.sh]="--single-gpu"
)

# Extra init delay (seconds) before polling — for scripts that need time before
# their internal processes are ready to accept connections
declare -A SCRIPT_INIT_DELAY=(
    [disagg_same_gpu.sh]=10
    [disagg_lmcache.sh]=20
)

# ---------------------------------------------------------------------------
# Globals set at runtime
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/test_logs/${TIMESTAMP}"
HARNESS_LOG="${LOG_DIR}/harness.log"
RESULTS_FILE="${SCRIPT_DIR}/test_results_${TIMESTAMP}.txt"

declare -A RESULT_STATE   # script → PASS|FAIL(...)|EXPECTED_FAIL|SKIPPED
declare -A RESULT_DURATION
declare -A RESULT_NOTE

CURRENT_PGID=""           # PGID of currently running test process group

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
ONLY_SCRIPT=""
START_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            ONLY_SCRIPT="$2"; shift 2 ;;
        --start-from)
            START_FROM="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--only <script>] [--start-from <script>]" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_ts() { date +%H:%M:%S; }

log() {
    local script="$1" level="$2"; shift 2
    local msg="[$(_ts)] [${script}] [${level}] $*"
    echo "$msg"
    echo "$msg" >> "${HARNESS_LOG}"
}

log_delimiter() {
    local line
    line="════════════════════════════════════════════════════════"
    echo "$line"
    echo "$line" >> "${HARNESS_LOG}"
}

# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

# Kill the current test's process group and sweep stragglers
cleanup_test() {
    local pgid="${1:-$CURRENT_PGID}"
    [[ -z "$pgid" ]] && return 0

    log "HARNESS" "CLEANUP" "Sending SIGTERM to process group -${pgid}"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    sleep 3

    log "HARNESS" "CLEANUP" "Sending SIGKILL to process group -${pgid}"
    kill -KILL -- "-${pgid}" 2>/dev/null || true

    # Named straggler sweep
    for pattern in "dynamo.frontend" "dynamo.vllm" "vllm serve" "vllm.entrypoints"; do
        pkill -f "${pattern}" 2>/dev/null || true
    done
    sleep 3
    for pattern in "dynamo.frontend" "dynamo.vllm" "vllm serve" "vllm.entrypoints"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done

    CURRENT_PGID=""

    # Wait for GPU to clear
    log "HARNESS" "CLEANUP" "Waiting for GPU compute processes to exit..."
    local elapsed=0
    while (( elapsed < GPU_DRAIN_TIMEOUT )); do
        local nproc
        nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if (( nproc == 0 )); then
            log "HARNESS" "CLEANUP" "GPU clear (0 compute apps)"
            break
        fi
        sleep 2
        (( elapsed += 2 ))
    done
    if (( elapsed >= GPU_DRAIN_TIMEOUT )); then
        log "HARNESS" "CLEANUP" "WARN: GPU still has processes after ${GPU_DRAIN_TIMEOUT}s"
    fi

    # Wait for ports to free
    log "HARNESS" "CLEANUP" "Waiting for ports to free..."
    elapsed=0
    while (( elapsed < PORT_DRAIN_TIMEOUT )); do
        local busy=0
        for port in "${ALL_PORTS[@]}"; do
            if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
                (( busy++ ))
            fi
        done
        if (( busy == 0 )); then
            log "HARNESS" "CLEANUP" "All ports free"
            break
        fi
        sleep 2
        (( elapsed += 2 ))
    done
    if (( elapsed >= PORT_DRAIN_TIMEOUT )); then
        log "HARNESS" "CLEANUP" "WARN: Some ports still in use after ${PORT_DRAIN_TIMEOUT}s"
    fi
}

# Trap handler: clean up and exit
cleanup_all() {
    log "HARNESS" "WARN" "Harness interrupted — cleaning up"
    cleanup_test "${CURRENT_PGID}"
    print_summary
}
trap 'cleanup_all; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Readiness polling
# ---------------------------------------------------------------------------
_models_ready() {
    # Returns 0 only when /v1/models responds with at least one model registered.
    # The frontend binds the port before any worker registers, so a bare HTTP-200
    # check causes false-positives that lead to 404 on /v1/chat/completions.
    local port="${1:-$BASE_PORT}"
    local body
    body=$(curl -sf "http://localhost:${port}/v1/models" 2>/dev/null) || return 1
    python3 -c "
import json, sys
d = json.loads(sys.argv[1])
exit(0 if d.get('data') else 1)
" "$body" 2>/dev/null
}

wait_for_ready() {
    local port="${1:-$BASE_PORT}"
    local elapsed=0
    while (( elapsed < STARTUP_TIMEOUT )); do
        if _models_ready "$port"; then
            return 0
        fi
        sleep "${HEALTH_POLL_INTERVAL}"
        (( elapsed += HEALTH_POLL_INTERVAL ))
    done
    return 1
}

# ---------------------------------------------------------------------------
# Request construction & validation
# ---------------------------------------------------------------------------

# Send one request; returns 0 on success, 1 on failure.
# Writes pass/fail details to HARNESS_LOG via the log() function.
send_request() {
    local script="$1" port="${2:-$BASE_PORT}" req_num="$3"
    local req_type="${SCRIPT_TYPE[$script]:-text}"
    local model="${SCRIPT_MODEL[$script]}"

    local body http_code response

    case "$req_type" in
        text)
            body=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
    'max_tokens': 32
}))" "$model")
            ;;
        multimodal)
            body=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': sys.argv[2]}},
            {'type': 'text', 'text': 'Describe this image briefly.'}
        ]
    }],
    'max_tokens': 64
}))" "$model" "$MULTIMODAL_IMAGE_URL")
            ;;
        omni_image|omni_video)
            body=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'max_tokens': 32
}))" "$model")
            ;;
    esac

    local tmpfile
    tmpfile=$(mktemp /tmp/dynamo_qa_resp.XXXXXX)

    http_code=$(curl -sf \
        -w "%{http_code}" \
        -o "${tmpfile}" \
        -X POST "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" \
        --max-time 60 \
        2>/dev/null) || http_code="000"

    response=$(cat "${tmpfile}" 2>/dev/null || echo "")
    rm -f "${tmpfile}"

    # Validate response
    local ok=0
    case "$req_type" in
        text|multimodal)
            ok=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    c = d.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(1 if isinstance(c, str) and len(c) > 0 else 0)
except Exception:
    print(0)
" "$response" 2>/dev/null || echo 0)
            ;;
        omni_image|omni_video)
            ok=$(python3 -c "
import json, sys
try:
    code = int(sys.argv[2])
    d = json.loads(sys.argv[1])
    print(1 if code == 200 and 'error' not in d else 0)
except Exception:
    print(0)
" "$response" "$http_code" 2>/dev/null || echo 0)
            ;;
    esac

    if [[ "$ok" == "1" ]]; then
        log "$script" "INFO" "  Request ${req_num}/${N_REQUESTS} on :${port} → OK (HTTP ${http_code})"
        return 0
    else
        log "$script" "FAIL" "  Request ${req_num}/${N_REQUESTS} on :${port} → FAILED (HTTP ${http_code})"
        log "$script" "FAIL" "  Response body: $(echo "$response" | head -c 500)"
        return 1
    fi
}

# Run N_REQUESTS against a port; return 0 if all pass
run_requests() {
    local script="$1" port="${2:-$BASE_PORT}"
    local failed=0
    for i in $(seq 1 "$N_REQUESTS"); do
        send_request "$script" "$port" "$i" || (( failed++ ))
    done
    return $failed
}

# ---------------------------------------------------------------------------
# Per-test tail-on-failure helper
# ---------------------------------------------------------------------------
show_log_tail() {
    local script="$1" logfile="$2"
    local lines=30
    log "$script" "FAIL" "── Last ${lines} lines of $(basename "$logfile") ──────────────"
    tail -n "$lines" "$logfile" 2>/dev/null | while IFS= read -r line; do
        log "$script" "FAIL" "  $line"
    done
    log "$script" "FAIL" "─────────────────────────────────────────────────"
}

# ---------------------------------------------------------------------------
# is_expected_failure / is_skip helper
# ---------------------------------------------------------------------------
is_in_array() {
    local needle="$1"; shift
    for el in "$@"; do
        [[ "$el" == "$needle" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Run a single test
# ---------------------------------------------------------------------------
run_one_test() {
    local script="$1" test_num="$2" total="$3"
    local script_path="${SCRIPT_DIR}/${script}"
    local logfile="${LOG_DIR}/${script%.sh}.log"
    local req_type="${SCRIPT_TYPE[$script]:-text}"
    local model="${SCRIPT_MODEL[$script]:-unknown}"
    local extra_args="${SCRIPT_EXTRA_ARGS[$script]:-}"
    local init_delay="${SCRIPT_INIT_DELAY[$script]:-0}"

    log_delimiter
    log "$script" "INFO" "▶ START TEST ${test_num}/${total}"
    log "$script" "INFO" "  Model:  ${model}"
    log "$script" "INFO" "  Type:   ${req_type}"
    log "$script" "INFO" "  Log:    test_logs/${TIMESTAMP}/${script%.sh}.log"
    [[ -n "$extra_args" ]] && log "$script" "INFO" "  Args:   ${extra_args}"
    log_delimiter

    local t_start
    t_start=$(date +%s)

    # ---------- SKIP ----------
    if is_in_array "$script" "${SKIP_SCRIPTS[@]}"; then
        log "$script" "INFO" "SKIPPED (hardware/topology not available)"
        RESULT_STATE[$script]="SKIPPED"
        RESULT_DURATION[$script]=0
        RESULT_NOTE[$script]="hardware skip"
        return 0
    fi

    # ---------- LAUNCH ----------
    log "$script" "INFO" "Launching: bash ${script} ${extra_args}"
    # shellcheck disable=SC2086
    setsid bash "${script_path}" ${extra_args} > "${logfile}" 2>&1 &
    local script_pid=$!
    CURRENT_PGID=$script_pid   # setsid makes PID == PGID

    # Watchdog: if the script process dies before we're done, note it
    local script_dead=0
    ( sleep 1
      while kill -0 "$script_pid" 2>/dev/null; do sleep 2; done
      # Write a sentinel so the parent can detect premature death
      echo "SCRIPT_DEAD" >> "${logfile}.sentinel"
    ) &
    local watchdog_pid=$!

    # ---------- WAIT FOR READY (with hard timeout) ----------
    if (( init_delay > 0 )); then
        log "$script" "INFO" "Waiting ${init_delay}s initial delay..."
        sleep "$init_delay"
    fi

    log "$script" "INFO" "Polling http://localhost:${BASE_PORT}/v1/models (up to ${STARTUP_TIMEOUT}s)..."

    local ready=0
    local elapsed=0
    local wall_start
    wall_start=$(date +%s)

    while (( elapsed < STARTUP_TIMEOUT )); do
        # Check hard wall-clock limit
        local wall_now
        wall_now=$(date +%s)
        if (( wall_now - wall_start > TOTAL_TEST_TIMEOUT )); then
            log "$script" "FAIL" "Hard wall-clock limit (${TOTAL_TEST_TIMEOUT}s) exceeded"
            break
        fi

        # Check if script died prematurely
        if [[ -f "${logfile}.sentinel" ]]; then
            log "$script" "FAIL" "Script process exited prematurely"
            script_dead=1
            break
        fi

        if _models_ready "${BASE_PORT}"; then
            ready=1
            break
        fi
        sleep "${HEALTH_POLL_INTERVAL}"
        (( elapsed += HEALTH_POLL_INTERVAL ))
    done

    rm -f "${logfile}.sentinel"
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    local t_ready
    t_ready=$(date +%s)

    # ---------- REQUESTS ----------
    local req_pass=0
    if (( ready == 1 )); then
        log "$script" "INFO" "Service ready after $(( t_ready - t_start ))s — sending requests"

        local req_failed=0
        run_requests "$script" "$BASE_PORT" || req_failed=$?

        # Special case: agg_router_replicas.sh also tests port 8001
        if [[ "$script" == "agg_router_replicas.sh" ]]; then
            log "$script" "INFO" "Waiting for replica port ${REPLICA_PORT}..."
            if wait_for_ready "${REPLICA_PORT}"; then
                run_requests "$script" "${REPLICA_PORT}" || (( req_failed++ ))
            else
                log "$script" "FAIL" "Replica port ${REPLICA_PORT} never became ready"
                (( req_failed++ ))
            fi
        fi

        if (( req_failed == 0 )); then
            req_pass=1
        fi
    fi

    # ---------- CLEANUP ----------
    log "$script" "CLEANUP" "Tearing down process group ${CURRENT_PGID}..."
    cleanup_test "${CURRENT_PGID}"

    local t_end
    t_end=$(date +%s)
    local duration=$(( t_end - t_start ))
    RESULT_DURATION[$script]=$duration

    # ---------- DETERMINE RESULT ----------
    local is_expected_fail=0
    is_in_array "$script" "${EXPECTED_FAILURES[@]}" && is_expected_fail=1

    if (( ready == 0 )); then
        if (( is_expected_fail )); then
            log "$script" "INFO" "EXPECTED_FAIL — service did not start (as expected)"
            RESULT_STATE[$script]="EXPECTED_FAIL"
            RESULT_NOTE[$script]="did not start"
        else
            log "$script" "FAIL" "Startup timed out after ${STARTUP_TIMEOUT}s"
            show_log_tail "$script" "$logfile"
            RESULT_STATE[$script]="FAIL(startup_timeout)"
            RESULT_NOTE[$script]="see ${script%.sh}.log"
        fi
    elif (( req_pass == 0 )); then
        if (( is_expected_fail )); then
            log "$script" "INFO" "EXPECTED_FAIL — requests failed (as expected)"
            RESULT_STATE[$script]="EXPECTED_FAIL"
            RESULT_NOTE[$script]="requests failed"
        else
            log "$script" "FAIL" "Service started but requests failed"
            show_log_tail "$script" "$logfile"
            RESULT_STATE[$script]="FAIL(requests)"
            RESULT_NOTE[$script]="see ${script%.sh}.log"
        fi
    else
        if (( is_expected_fail )); then
            log "$script" "WARN" "UNEXPECTED_PASS — expected failure but all requests passed"
            RESULT_STATE[$script]="UNEXPECTED_PASS"
            RESULT_NOTE[$script]="expected to fail"
        else
            log "$script" "PASS" "All ${N_REQUESTS} requests passed (${duration}s)"
            RESULT_STATE[$script]="PASS"
            RESULT_NOTE[$script]=""
        fi
    fi

    log "$script" "INFO" "Duration: ${duration}s"
    sleep "${BETWEEN_TEST_SLEEP}"
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print_summary() {
    local line
    local header="| # | Script                             | Result               | Duration | Note                            |"
    local sep="|---|------------------------------------|-----------------------|----------|---------------------------------|"

    {
        echo ""
        echo "## QA Run Results — ${TIMESTAMP}"
        echo ""
        echo "$header"
        echo "$sep"

        local i=0
        for script in "${TEST_SCRIPTS[@]}"; do
            (( i++ ))
            local state="${RESULT_STATE[$script]:-NOT_RUN}"
            local dur="${RESULT_DURATION[$script]:-0}"
            local note="${RESULT_NOTE[$script]:-}"
            printf "| %-2d | %-34s | %-21s | %-8s | %-31s |\n" \
                "$i" "$script" "$state" "${dur}s" "$note"
        done
        echo ""
    } | tee -a "${HARNESS_LOG}" | tee "${RESULTS_FILE}"

    echo ""
    echo "Results saved to: ${RESULTS_FILE}"
    echo "Logs in:          ${LOG_DIR}/"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    mkdir -p "${LOG_DIR}"
    touch "${HARNESS_LOG}"

    log "HARNESS" "INFO" "QA harness starting — ${TIMESTAMP}"
    log "HARNESS" "INFO" "Log dir: ${LOG_DIR}"
    log "HARNESS" "INFO" "Results file: ${RESULTS_FILE}"

    # Pre-mark skipped scripts
    for script in "${SKIP_SCRIPTS[@]}"; do
        RESULT_STATE[$script]="SKIPPED"
        RESULT_DURATION[$script]=0
        RESULT_NOTE[$script]="hardware skip"
    done

    # Determine which scripts to run
    local scripts_to_run=()
    if [[ -n "$ONLY_SCRIPT" ]]; then
        scripts_to_run=("$ONLY_SCRIPT")
    else
        local skip_mode=0
        [[ -n "$START_FROM" ]] && skip_mode=1

        for script in "${TEST_SCRIPTS[@]}"; do
            if (( skip_mode )) && [[ "$script" == "$START_FROM" ]]; then
                skip_mode=0
            fi
            if (( skip_mode == 0 )); then
                scripts_to_run+=("$script")
            fi
        done

        if [[ -n "$START_FROM" ]] && (( ${#scripts_to_run[@]} == 0 )); then
            echo "ERROR: --start-from script '${START_FROM}' not found in test list" >&2
            exit 1
        fi
    fi

    local total=${#scripts_to_run[@]}
    log "HARNESS" "INFO" "Running ${total} tests"

    local test_num=0
    for script in "${scripts_to_run[@]}"; do
        (( test_num++ ))

        if ! [[ -f "${SCRIPT_DIR}/${script}" ]]; then
            log "$script" "WARN" "Script not found: ${SCRIPT_DIR}/${script} — skipping"
            RESULT_STATE[$script]="SKIPPED"
            RESULT_DURATION[$script]=0
            RESULT_NOTE[$script]="file not found"
            continue
        fi

        run_one_test "$script" "$test_num" "$total"
    done

    log_delimiter
    log "HARNESS" "INFO" "All tests complete"
    print_summary
}

main "$@"
