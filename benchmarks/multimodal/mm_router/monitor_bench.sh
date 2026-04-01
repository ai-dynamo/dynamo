#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained benchmark monitor.
# Runs inside the container alongside run_bench_all.sh.
# Watches for FAILED scenarios, diagnoses, fixes, and signals retries.
# Writes a report to logs/monitor_report.txt when done.

BENCH_DIR=/workspace/benchmarks/multimodal/mm_router
LOG_DIR="${BENCH_DIR}/logs"
REPORT="${LOG_DIR}/monitor_report.txt"
SCENARIOS=(rr mm vllm-processor rust)

mkdir -p "${LOG_DIR}"

log() {
    local msg="[monitor $(date '+%H:%M:%S')] $*"
    echo "${msg}" | tee -a "${REPORT}"
}

fix_stuck_processes() {
    log "Killing stuck dynamo/vllm processes..."
    pkill -9 -f 'dynamo.vllm' 2>/dev/null || true
    pkill -9 -f 'dynamo.frontend' 2>/dev/null || true
    pkill -9 -f 'mm_router_worker' 2>/dev/null || true
    sleep 10
    log "Killed. Waiting 10s for GPU memory to free..."
}

diagnose_and_fix() {
    local name="$1"
    local logfile="${LOG_DIR}/run_${name}.log"

    log "=== Diagnosing failure: ${name} ==="

    # Capture last 50 lines of run log
    local last_lines
    last_lines=$(tail -50 "${logfile}" 2>/dev/null || echo "(no log)")
    log "--- Last 50 lines of run_${name}.log ---"
    echo "${last_lines}" >> "${REPORT}"

    # GPU memory
    log "--- GPU memory ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null >> "${REPORT}" || true

    # Running processes
    log "--- dynamo/vllm processes ---"
    ps aux | grep -E 'dynamo|vllm' | grep -v grep >> "${REPORT}" || true

    # Port conflicts
    log "--- Ports 8000/18081/18083 in use ---"
    ss -tlnp 2>/dev/null | grep -E ':8000|:18081|:18083' >> "${REPORT}" || true

    # Determine fix
    local fixed=0

    if echo "${last_lines}" | grep -qiE 'address already in use|port.*in use|bind.*failed'; then
        log "Root cause: PORT CONFLICT. Killing stuck processes."
        fix_stuck_processes
        fixed=1
    elif echo "${last_lines}" | grep -qiE 'cuda out of memory|out of memory|oom'; then
        log "Root cause: OOM. Killing processes and waiting 30s."
        fix_stuck_processes
        sleep 20
        fixed=1
    elif echo "${last_lines}" | grep -qiE 'connection refused|nats|etcd'; then
        log "Root cause: NATS/etcd connectivity. Attempting restart."
        cd /workspace && docker compose -f deploy/docker-compose.yml up -d 2>/dev/null || true
        sleep 10
        fixed=1
    else
        log "Root cause: UNKNOWN. Applying generic fix (kill processes)."
        fix_stuck_processes
        fixed=1
    fi

    if [[ $fixed -eq 1 ]]; then
        log "Fix applied. Writing retry signal for ${name}."
        echo retry > "${LOG_DIR}/${name}.retry"
    fi
}

log "=== Monitor started. Watching scenarios: ${SCENARIOS[*]} ==="

REPORTED_ISSUES=()

while true; do
    # Check if all done
    if [[ -f "${LOG_DIR}/bench.status" ]] && grep -q "ALL_DONE" "${LOG_DIR}/bench.status" 2>/dev/null; then
        log "bench.status = ALL_DONE. Monitoring complete."
        break
    fi

    # Check each scenario for FAILED
    for name in "${SCENARIOS[@]}"; do
        status_file="${LOG_DIR}/${name}.status"
        if [[ ! -f "${status_file}" ]]; then
            continue
        fi

        status=$(cat "${status_file}" 2>/dev/null)

        if [[ "${status}" == "FAILED" ]]; then
            # Only act if no retry is already pending and we haven't recently fixed this
            if [[ ! -f "${LOG_DIR}/${name}.retry" ]]; then
                local already_reported=0
                for issue in "${REPORTED_ISSUES[@]:-}"; do
                    if [[ "${issue}" == "${name}" ]]; then
                        already_reported=1
                        break
                    fi
                done

                if [[ $already_reported -eq 0 ]]; then
                    REPORTED_ISSUES+=("${name}")
                    diagnose_and_fix "${name}"
                fi
            fi
        fi
    done

    sleep 60
done

# Final summary
log ""
log "=== FINAL SCENARIO STATUS ==="
for name in "${SCENARIOS[@]}"; do
    status=$(cat "${LOG_DIR}/${name}.status" 2>/dev/null || echo "UNKNOWN")
    log "  ${name}: ${status}"
done

log ""
log "=== ISSUES RESOLVED ==="
if [[ ${#REPORTED_ISSUES[@]} -eq 0 ]]; then
    log "  No issues detected."
else
    for name in "${REPORTED_ISSUES[@]}"; do
        final_status=$(cat "${LOG_DIR}/${name}.status" 2>/dev/null || echo "UNKNOWN")
        log "  ${name}: diagnosed+fixed -> final status: ${final_status}"
    done
fi

log ""
log "Monitor report complete. See ${REPORT} for full details."
log "Run parse_results.py to generate the results table."
