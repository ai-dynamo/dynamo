#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BPF script runner with capability detection.
#
# Usage:
#   ./run.sh --pid 12345 runqlat             # run specific script
#   ./run.sh --pid 12345 --batch --output-dir /tmp/bpf --duration 30
#   ./run.sh --list                          # list available scripts
#   ./run.sh --check                         # check capabilities

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID=""
DURATION="30"
BATCH=false
OUTPUT_DIR=""

# Default set of scripts for batch mode (safe subset that works with --pid)
BATCH_SCRIPTS=(runqlat syscall_latency transport_latency context_switches offcputime cpudist)

# Available scripts
SCRIPTS=(
    "runqlat:CPU run queue latency"
    "cpudist:On-CPU time distribution"
    "offcputime:Off-CPU stack traces"
    "funclatency:Function latency (template)"
    "transport_latency:Socket read/write latency"
    "tcplife:TCP connection lifetimes"
    "tcpretrans:TCP retransmissions"
    "syscall_latency:Top slow syscalls"
    "context_switches:Context switch histograms"
)

check_capabilities() {
    echo "=== BPF Capability Check ==="
    local ok=true

    if ! command -v bpftrace &>/dev/null; then
        echo "FAIL: bpftrace not found (install: apt install bpftrace)"
        ok=false
    else
        echo "OK:   bpftrace $(bpftrace --version 2>/dev/null | head -1)"
    fi

    if [[ $(id -u) -ne 0 ]]; then
        echo "WARN: Not running as root. BPF scripts require CAP_BPF + CAP_PERFMON."
        # Check effective capabilities from the "Current:" line.
        # The IAB line uses !cap_xxx for *denied* caps — grepping the full
        # output would false-positive on those negated entries.
        if command -v capsh &>/dev/null; then
            local current_caps
            current_caps=$(capsh --print 2>/dev/null | grep '^Current:' || true)
            if echo "$current_caps" | grep -q cap_bpf; then
                echo "OK:   CAP_BPF available"
            else
                echo "FAIL: CAP_BPF not available"
                ok=false
            fi
            if echo "$current_caps" | grep -q cap_perfmon; then
                echo "OK:   CAP_PERFMON available"
            else
                echo "FAIL: CAP_PERFMON not available"
                ok=false
            fi
        else
            echo "WARN: capsh not found, cannot check capabilities"
        fi
    else
        echo "OK:   Running as root"
    fi

    local kernel_ver
    kernel_ver=$(uname -r | cut -d. -f1-2)
    local major minor
    major=$(echo "$kernel_ver" | cut -d. -f1)
    minor=$(echo "$kernel_ver" | cut -d. -f2)
    if [[ $major -gt 4 ]] || { [[ $major -eq 4 ]] && [[ $minor -ge 18 ]]; }; then
        echo "OK:   Kernel $(uname -r) (>= 4.18 required)"
    else
        echo "FAIL: Kernel $(uname -r) (>= 4.18 required)"
        ok=false
    fi

    if [[ "$ok" == true ]]; then
        echo ""
        echo "All checks passed. Ready to trace."
    else
        echo ""
        echo "Some checks failed. Fix issues above before tracing."
        return 1
    fi
}

list_scripts() {
    echo "Available BPF scripts:"
    echo ""
    for entry in "${SCRIPTS[@]}"; do
        local name="${entry%%:*}"
        local desc="${entry#*:}"
        printf "  %-25s %s\n" "$name" "$desc"
    done
}

run_script() {
    local name=$1
    local script="${SCRIPT_DIR}/traces/${name}.bt"

    if [[ ! -f "$script" ]]; then
        echo "ERROR: Script not found: $script"
        echo "Available scripts:"
        list_scripts
        return 1
    fi

    local args=()
    if [[ -n "$PID" ]]; then
        args+=(-p "$PID")
    fi

    echo "Running: bpftrace ${args[*]} $script"
    echo "Press Ctrl-C to stop."
    echo ""
    exec bpftrace "${args[@]}" "$script"
}

run_batch() {
    # Run multiple BPF scripts in parallel, capturing output to files.
    # Called by run_perf.sh as a single background job.
    # This function waits for all children internally and handles signal
    # forwarding (TERM→INT) so bpftrace flushes its aggregation maps.
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: --batch requires --output-dir" >&2
        exit 1
    fi
    if [[ -z "$PID" ]]; then
        echo "ERROR: --batch requires --pid" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    local pids=()

    # Forward SIGTERM/SIGINT to all bpftrace children as SIGINT so they flush.
    # Then wait for them to finish writing output before exiting.
    trap 'for p in "${pids[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait 2>/dev/null; exit 0' TERM INT

    for script_name in "${BATCH_SCRIPTS[@]}"; do
        local bt_file="${SCRIPT_DIR}/traces/${script_name}.bt"
        if [[ ! -f "$bt_file" ]]; then
            echo "  [bpf] SKIP $script_name (not found)" >&2
            continue
        fi
        echo "  [bpf] $script_name (duration=${DURATION}s)" >&2
        # SIGINT (not SIGTERM) so bpftrace flushes aggregation maps before exiting
        timeout --signal=INT "$DURATION" bpftrace -p "$PID" "$bt_file" \
            > "$OUTPUT_DIR/${script_name}.txt" 2>&1 &
        pids+=($!)
    done

    # Also run system-wide scripts (no --pid) that are safe
    for script_name in tcplife tcpretrans; do
        local bt_file="${SCRIPT_DIR}/traces/${script_name}.bt"
        if [[ -f "$bt_file" ]]; then
            echo "  [bpf] $script_name (system-wide, duration=${DURATION}s)" >&2
            timeout --signal=INT "$DURATION" bpftrace "$bt_file" \
                > "$OUTPUT_DIR/${script_name}.txt" 2>&1 &
            pids+=($!)
        fi
    done

    # Wait for all bpftrace processes to complete (timeout or signal).
    # This keeps run.sh alive so the caller can kill us to stop everything.
    for p in "${pids[@]}"; do
        wait "$p" 2>/dev/null || true
    done
    trap - TERM INT
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pid|-p)        PID="$2"; shift 2 ;;
        --duration|-d)   DURATION="$2"; shift 2 ;;
        --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
        --batch)         BATCH=true; shift ;;
        --check)         check_capabilities; exit $? ;;
        --list|-l)       list_scripts; exit 0 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [SCRIPT_NAME]"
            echo ""
            echo "Single script mode:"
            echo "  $0 --pid PID SCRIPT_NAME"
            echo ""
            echo "Batch mode (for capture integration):"
            echo "  $0 --batch --pid PID --output-dir DIR [--duration SECS]"
            echo ""
            echo "Options:"
            echo "  --pid PID          Attach to specific process"
            echo "  --batch            Run all scripts in parallel"
            echo "  --output-dir DIR   Write output files to DIR (batch mode)"
            echo "  --duration SECS    Timeout per script (default: 30)"
            echo "  --check            Check BPF capabilities"
            echo "  --list             List available scripts"
            echo ""
            list_scripts
            exit 0
            ;;
        *)  break ;;
    esac
done

if [[ "$BATCH" == true ]]; then
    run_batch
    exit 0
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: No script specified."
    echo ""
    list_scripts
    exit 1
fi

run_script "$1"
