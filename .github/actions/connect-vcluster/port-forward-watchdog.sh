#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep the vCluster API port-forward on 127.0.0.1:8443 available to later steps.
#
# Usage: port-forward-watchdog.sh <pid-file> <namespace> <vcluster-name>
#
# <pid-file> holds the PID of the current `kubectl port-forward`; the watchdog
# rewrites it whenever it starts a replacement. Its log lines and the output of
# the port-forwards it starts go to stdout.
#
# Every INTERVAL (3s) the watchdog requests /healthz through the tunnel; any
# HTTP response counts as success. After a failed probe it restarts the
# port-forward:
# - at once if the process has exited. kubectl exits when its tunnel breaks
#   ("lost connection to pod"), so this is how a broken tunnel usually shows.
# - only after FAILURE_THRESHOLD (3) consecutive failures if the process still
#   runs, because killing it drops every request in flight through the tunnel.
# A probe may take PROBE_TIMEOUT (10s), as long as client-go (kubectl, Helm)
# allows for a TLS handshake. A single probe with a 1s connect timeout, which
# covers the TLS handshake too, used to kill working tunnels while Helm loaded
# them. Now a running port-forward is replaced only after the API has been
# unreachable for at least 6s (every probe failing at once) or 36s (every probe
# timing out).
#
# Including about 2s for kubectl to listen again, the tunnel is back at most 5s
# after kubectl exits, 12s after a running kubectl starts failing every probe at
# once, and 42s after it starts hanging every probe. Consumers retry connection
# errors for about 15s (.github/scripts/retry_kubectl.sh,
# tests/deploy/vcluster_utils.py), which covers the first two cases; against a
# hanging tunnel each of their attempts first waits out its own 10s TLS
# handshake timeout.
#
# The VCLUSTER_WATCHDOG_* variables override these defaults for tests.

interval=${VCLUSTER_WATCHDOG_INTERVAL:-3}
probe_timeout=${VCLUSTER_WATCHDOG_PROBE_TIMEOUT:-10}
failure_threshold=${VCLUSTER_WATCHDOG_FAILURE_THRESHOLD:-3}
stop_timeout=${VCLUSTER_WATCHDOG_STOP_TIMEOUT:-10}

log() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

# kill -0 also succeeds for a zombie. The port-forward started by connect-vcluster
# is not a child of this script, so after it exits it can stay a zombie until
# its new parent reaps it. Linux reports the state in /proc even without ps.
running() {
    local state
    [[ $1 =~ ^[1-9][0-9]*$ ]] && kill -0 "$1" 2>/dev/null || return 1
    if [[ -d /proc/self ]]; then
        { read -r state <"/proc/$1/stat"; } 2>/dev/null || return 1
        state=${state##*) }
    else
        state=$(ps -o stat= -p "$1" 2>/dev/null)
    fi
    [[ ${state} != Z* ]]
}

probe() {
    probe_error=$(curl -sSk --max-time "${probe_timeout}" -o /dev/null \
        https://127.0.0.1:8443/healthz 2>&1)
}

# Wait up to $2 seconds for process $1 to exit.
wait_for_exit() {
    local remaining=$2
    while running "$1"; do
        if ((remaining-- <= 0)); then
            return 1
        fi
        sleep 1
    done
}

# The kernel closes the listener on 127.0.0.1:8443 when the process exits, so
# a replacement can bind the port once this returns.
stop_port_forward() {
    kill "$1" 2>/dev/null
    wait_for_exit "$1" "${stop_timeout}" && return
    log "Port-forward pid $1 still running ${stop_timeout}s after SIGTERM; sending SIGKILL"
    kill -KILL "$1" 2>/dev/null
    wait_for_exit "$1" "${stop_timeout}" ||
        log "Port-forward pid $1 still running ${stop_timeout}s after SIGKILL; starting a replacement anyway"
}

start_port_forward() {
    log "Watchdog restarting vCluster API port-forward: $1"
    kubectl port-forward -n "${namespace}" "svc/${vcluster_name}" 8443:443 &
    echo "$!" >"${pid_file}"
    log "Started vCluster API port-forward pid $!"
}

main() {
    if (($# != 3)); then
        echo "usage: $0 <pid-file> <namespace> <vcluster-name>" >&2
        exit 2
    fi
    pid_file=$1
    namespace=$2
    vcluster_name=$3
    log "Watchdog started: probing every ${interval}s with a ${probe_timeout}s timeout; restarting the" \
        "port-forward when it exits or after ${failure_threshold} consecutive failed probes"

    local pid watched_pid="" failures=0
    while true; do
        pid=$(cat "${pid_file}" 2>/dev/null)
        # Count failures per process, so that a new port-forward starts afresh.
        if [[ ${pid} != "${watched_pid}" ]]; then
            watched_pid=${pid}
            failures=0
        fi
        if probe; then
            if ((failures > 0)); then
                log "Probe succeeded after ${failures} consecutive failed probe(s)"
                failures=0
            fi
        elif ! running "${pid}"; then
            start_port_forward "pid ${pid:-unknown} exited"
        else
            failures=$((failures + 1))
            log "Probe failed (${failures}/${failure_threshold}) with port-forward pid ${pid} running: ${probe_error}"
            if ((failures >= failure_threshold)); then
                stop_port_forward "${pid}"
                start_port_forward "${failures} consecutive failed probes with pid ${pid} running"
            fi
        fi
        sleep "${interval}"
    done
}

# Bash reads a script while running it. Calling main last makes it parse the
# whole file first, so the endless loop does not depend on the file afterwards.
main "$@"
