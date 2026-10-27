#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

pf_pid_file="${PF_PID_FILE:-}"
pf_log="${PF_LOG:-/dev/null}"
watchdog_log="${WATCHDOG_LOG:-${pf_log}}"
namespace="${NAMESPACE:-}"
vcluster_name="${VCLUSTER_NAME:-}"
failure_threshold="${FAILURE_THRESHOLD:-3}"
stop_wait_attempts="${STOP_WAIT_ATTEMPTS:-10}"
port_release_attempts="${PORT_RELEASE_ATTEMPTS:-10}"
readiness_attempts="${READINESS_ATTEMPTS:-0}"
readiness_timeout="${READINESS_TIMEOUT:-60}"
wait_interval="${WAIT_INTERVAL:-1}"
watchdog_interval="${WATCHDOG_INTERVAL:-5}"
watchdog_pid_file="${WATCHDOG_PID_FILE:-}"
watchdog_lock_file="${WATCHDOG_LOCK_FILE:-}"
failure_count=0
candidate_pid=""
candidate_log_start=0
shutdown_requested=0

log() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "${watchdog_log}"
}

valid_pid() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 1 ]
}

read_pid_file() {
    local pid
    [ -f "${pf_pid_file}" ] || return 1
    IFS= read -r pid < "${pf_pid_file}" || true
    valid_pid "${pid}" || return 1
    printf '%s\n' "${pid}"
}

process_is_port_forward() {
    local pid command
    pid="${1:-}"
    valid_pid "${pid}" || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    command="$(ps -p "${pid}" -o command= 2>/dev/null)" || return 1
    case "${command}" in
        "kubectl port-forward -n ${namespace} svc/${vcluster_name} 8443:443"*|\
        *"/kubectl port-forward -n ${namespace} svc/${vcluster_name} 8443:443"*) return 0 ;;
        *) return 1 ;;
    esac
}

process_has_exited() {
    local pid state
    pid="$1"
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi
    state="$(ps -p "${pid}" -o stat= 2>/dev/null)" || return 0
    case "${state}" in
        *Z*) return 0 ;;
        *) return 1 ;;
    esac
}

record_probe_result() {
    if [ "${1:-1}" -eq 0 ]; then
        failure_count=0
        return 1
    fi

    failure_count=$((failure_count + 1))
    [ "${failure_count}" -ge "${failure_threshold}" ]
}

pid_file_still_matches() {
    local current
    current="$(read_pid_file)" || return 1
    [ "${current}" = "$1" ]
}

remove_pid_file_if_matches() {
    local observed current
    observed="$1"
    [ -f "${pf_pid_file}" ] || return 0
    IFS= read -r current < "${pf_pid_file}" || true
    [ "${current}" = "${observed}" ] || return 1
    rm -f "${pf_pid_file}"
}

wait_for_process_exit() {
    local pid attempt
    pid="$1"
    attempt=0
    while [ "${attempt}" -lt "${stop_wait_attempts}" ]; do
        if process_has_exited "${pid}"; then
            return 0
        fi
        sleep "${wait_interval}"
        attempt=$((attempt + 1))
    done
    return 1
}

stop_port_forward() {
    local observed_pid
    observed_pid="$1"

    if ! pid_file_still_matches "${observed_pid}"; then
        log "PID file changed before stop; leaving observed pid ${observed_pid} untouched"
        return 1
    fi
    if ! process_is_port_forward "${observed_pid}"; then
        log "PID ${observed_pid} is not the expected kubectl port-forward"
        return 1
    fi

    kill -TERM "${observed_pid}" 2>/dev/null || true
    if wait_for_process_exit "${observed_pid}"; then
        return 0
    fi

    if ! pid_file_still_matches "${observed_pid}" ||
        ! process_is_port_forward "${observed_pid}"; then
        log "PID identity changed before SIGKILL; refusing escalation"
        return 1
    fi

    kill -KILL "${observed_pid}" 2>/dev/null || true
    wait_for_process_exit "${observed_pid}"
}

probe_health() {
    local max_time
    max_time="${1:-${PROBE_MAX_TIME:-5}}"
    curl -fsk \
        --connect-timeout "${PROBE_CONNECT_TIMEOUT:-3}" \
        --max-time "${max_time}" \
        https://127.0.0.1:8443/healthz >/dev/null
}

port_is_free() {
    local status
    curl -sk \
        --connect-timeout "${LISTENER_CONNECT_TIMEOUT:-1}" \
        --max-time "${LISTENER_MAX_TIME:-1}" \
        https://127.0.0.1:8443/healthz >/dev/null
    status=$?
    [ "${status}" -eq 7 ]
}

wait_for_port_release() {
    local attempt
    attempt=0
    while [ "${attempt}" -lt "${port_release_attempts}" ]; do
        if port_is_free; then
            return 0
        fi
        sleep "${wait_interval}"
        attempt=$((attempt + 1))
    done
    return 1
}

candidate_has_address_conflict() {
    tail -c "+$((candidate_log_start + 1))" "${pf_log}" 2>/dev/null |
        grep -qi 'address already in use'
}

start_candidate() {
    mkdir -p "$(dirname "${pf_log}")"
    touch "${pf_log}"
    candidate_log_start="$(wc -c < "${pf_log}" | tr -d ' ')"
    printf '\n[%s] Watchdog starting vCluster API port-forward candidate\n' \
        "$(date -u +%FT%TZ)" >> "${pf_log}"
    candidate_log_start="$(wc -c < "${pf_log}" | tr -d ' ')"
    kubectl port-forward -n "${namespace}" "svc/${vcluster_name}" 8443:443 \
        >> "${pf_log}" 2>&1 &
    candidate_pid=$!
    log "Started candidate pid ${candidate_pid}"
}

wait_for_candidate_ready() {
    local pid attempt deadline healthy_checks remaining
    pid="$1"
    candidate_log_start="${2:-${candidate_log_start}}"
    attempt=0
    deadline=$((SECONDS + readiness_timeout))
    healthy_checks=0

    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if [ "${readiness_attempts}" -gt 0 ] &&
            [ "${attempt}" -ge "${readiness_attempts}" ]; then
            break
        fi
        if ! process_is_port_forward "${pid}"; then
            log "Candidate pid ${pid} exited before readiness"
            return 1
        fi
        if candidate_has_address_conflict; then
            log "Candidate pid ${pid} could not bind port 8443"
            return 1
        fi
        remaining=$((deadline - SECONDS))
        if [ "${remaining}" -le 0 ]; then
            break
        fi
        if probe_health "$((remaining < 5 ? remaining : 5))"; then
            if process_is_port_forward "${pid}" &&
                ! candidate_has_address_conflict; then
                healthy_checks=$((healthy_checks + 1))
                if [ "${healthy_checks}" -ge 2 ]; then
                    return 0
                fi
            else
                log "Health succeeded after candidate pid ${pid} exited"
                return 1
            fi
        else
            healthy_checks=0
        fi
        sleep "${wait_interval}"
        attempt=$((attempt + 1))
    done

    log "Candidate pid ${pid} did not become ready"
    return 1
}

cleanup_candidate() {
    local pid
    pid="$1"
    if process_is_port_forward "${pid}"; then
        kill -TERM "${pid}" 2>/dev/null || true
        if ! wait_for_process_exit "${pid}" &&
            process_is_port_forward "${pid}"; then
            kill -KILL "${pid}" 2>/dev/null || true
            wait_for_process_exit "${pid}" || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
}

publish_pid_atomically() {
    publish_pid_file_atomically "${pf_pid_file}" "$1"
}

publish_pid_file_atomically() {
    local destination pid temporary
    destination="$1"
    pid="$2"
    temporary="${destination}.tmp.$$"
    printf '%s\n' "${pid}" > "${temporary}" || return 1
    if ! mv "${temporary}" "${destination}"; then
        rm -f "${temporary}"
        return 1
    fi
}

replace_port_forward() {
    start_candidate
    if ! wait_for_candidate_ready "${candidate_pid}" "${candidate_log_start}"; then
        cleanup_candidate "${candidate_pid}"
        candidate_pid=""
        return 1
    fi
    if ! process_is_port_forward "${candidate_pid}" ||
        candidate_has_address_conflict; then
        log "Candidate pid ${candidate_pid} exited before publication"
        cleanup_candidate "${candidate_pid}"
        candidate_pid=""
        return 1
    fi
    if ! publish_pid_atomically "${candidate_pid}"; then
        cleanup_candidate "${candidate_pid}"
        candidate_pid=""
        return 1
    fi
    log "Published ready candidate pid ${candidate_pid}"
}

recover_unmanaged_port_forward() {
    local health_status
    probe_health
    health_status=$?
    if [ "${health_status}" -eq 0 ]; then
        failure_count=0
        log "Healthy unowned listener detected; leaving it unchanged"
        return 0
    fi

    if [ "${health_status}" -ne 7 ] || ! port_is_free; then
        log "Health failed but port 8443 still has an unowned listener; retrying"
        return 1
    fi

    replace_port_forward
}

watchdog_iteration() {
    local observed_pid probe_status

    if observed_pid="$(read_pid_file)" &&
        process_is_port_forward "${observed_pid}"; then
        if probe_health; then
            probe_status=0
        else
            probe_status=$?
        fi
        if ! record_probe_result "${probe_status}"; then
            if [ "${probe_status}" -ne 0 ]; then
                log "Health probe failed with status ${probe_status}; consecutive failures ${failure_count}/${failure_threshold}"
            fi
            return 0
        fi

        log "Health probe failed ${failure_count} consecutive times; replacing pid ${observed_pid}"
        failure_count=0
        if ! stop_port_forward "${observed_pid}"; then
            return 1
        fi
        remove_pid_file_if_matches "${observed_pid}" || return 1
        if [ "${candidate_pid}" = "${observed_pid}" ]; then
            wait "${observed_pid}" 2>/dev/null || true
            candidate_pid=""
        fi
        if ! wait_for_port_release; then
            log "Port 8443 was not released; refusing to start another tunnel"
            return 1
        fi
        replace_port_forward
        return
    fi

    failure_count=0
    if [ -n "${observed_pid:-}" ]; then
        log "PID ${observed_pid} is gone or no longer the expected port-forward"
        remove_pid_file_if_matches "${observed_pid}" || true
    elif [ -f "${pf_pid_file}" ]; then
        log "Ignoring invalid port-forward PID file"
        rm -f "${pf_pid_file}"
    fi
    recover_unmanaged_port_forward
}

process_is_watchdog() {
    local pid command
    pid="$1"
    valid_pid "${pid}" || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    command="$(ps -p "${pid}" -o command= 2>/dev/null)" || return 1
    case "${command}" in
        *"/watchdog.sh run"*) return 0 ;;
        *) return 1 ;;
    esac
}

watchdog_pid_file_still_matches() {
    local observed current
    observed="$1"
    [ -f "${watchdog_pid_file}" ] || return 1
    IFS= read -r current < "${watchdog_pid_file}" || true
    [ "${current}" = "${observed}" ]
}

remove_watchdog_pid_if_matches() {
    watchdog_pid_file_still_matches "$1" || return 1
    rm -f "${watchdog_pid_file}"
}

stop_watchdog_process() {
    local pid
    [ -f "${watchdog_pid_file}" ] || return 0
    IFS= read -r pid < "${watchdog_pid_file}" || true
    if ! valid_pid "${pid}" || ! process_is_watchdog "${pid}"; then
        log "Ignoring stale or invalid watchdog PID file"
        rm -f "${watchdog_pid_file}"
        return 0
    fi

    watchdog_pid_file_still_matches "${pid}" || return 1
    kill -TERM "${pid}" 2>/dev/null || true
    if wait_for_process_exit "${pid}"; then
        remove_watchdog_pid_if_matches "${pid}" || true
        return 0
    fi
    if ! watchdog_pid_file_still_matches "${pid}" ||
        ! process_is_watchdog "${pid}"; then
        return 1
    fi
    kill -KILL "${pid}" 2>/dev/null || true
    wait_for_process_exit "${pid}" || true
    remove_watchdog_pid_if_matches "${pid}" || true
}

published_watchdog_is_running() {
    local pid
    [ -f "${watchdog_pid_file}" ] || return 1
    IFS= read -r pid < "${watchdog_pid_file}" || true
    valid_pid "${pid}" && process_is_watchdog "${pid}"
}

stop_published_port_forward() {
    local pid
    if ! pid="$(read_pid_file)"; then
        [ ! -f "${pf_pid_file}" ] || rm -f "${pf_pid_file}"
        return 0
    fi
    if ! process_is_port_forward "${pid}"; then
        log "Ignoring stale port-forward PID ${pid}"
        remove_pid_file_if_matches "${pid}" || true
        return 0
    fi
    stop_port_forward "${pid}" || return 1
    remove_pid_file_if_matches "${pid}" || true
}

on_watchdog_exit() {
    if [ "${shutdown_requested}" -eq 1 ] &&
        [ -n "${candidate_pid}" ] &&
        pid_file_still_matches "${candidate_pid}" &&
        process_is_port_forward "${candidate_pid}"; then
        stop_port_forward "${candidate_pid}" || true
        remove_pid_file_if_matches "${candidate_pid}" || true
        wait "${candidate_pid}" 2>/dev/null || true
    fi
    remove_watchdog_pid_if_matches "$$" || true
}

run_watchdog() {
    local iterations max_iterations
    max_iterations="${WATCHDOG_MAX_ITERATIONS:-0}"
    iterations=0

    exec 9>"${watchdog_lock_file}"
    if ! flock -n 9; then
        log "Another vCluster port-forward watchdog already owns the lock"
        return 0
    fi

    if ! publish_pid_file_atomically "${watchdog_pid_file}" "$$"; then
        log "Failed to publish watchdog pid $$"
        return 1
    fi
    trap 'shutdown_requested=1; exit 0' TERM INT
    trap on_watchdog_exit EXIT
    log "Watchdog started with pid $$"

    while :; do
        watchdog_iteration || true
        iterations=$((iterations + 1))
        if [ "${max_iterations}" -gt 0 ] &&
            [ "${iterations}" -ge "${max_iterations}" ]; then
            break
        fi
        sleep "${watchdog_interval}"
    done
}

main() {
    case "${1:-}" in
        run)
            run_watchdog
            ;;
        stop-watchdog)
            stop_watchdog_process
            ;;
        stop-port-forward)
            stop_published_port_forward
            ;;
        status-watchdog)
            published_watchdog_is_running
            ;;
        *)
            echo "usage: $0 {run|status-watchdog|stop-watchdog|stop-port-forward}" >&2
            return 2
            ;;
    esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
