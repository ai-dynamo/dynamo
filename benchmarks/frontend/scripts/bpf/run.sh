#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BPF script runner with capability detection.
#
# Usage:
#   ./run.sh                          # run all scripts (requires root)
#   ./run.sh runqlat                  # run specific script
#   ./run.sh --pid 12345 runqlat      # attach to specific PID
#   ./run.sh --list                   # list available scripts
#   ./run.sh --check                  # check capabilities

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID=""
DURATION=""

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
        # Check for specific capabilities
        if command -v capsh &>/dev/null; then
            if capsh --print 2>/dev/null | grep -q cap_bpf; then
                echo "OK:   CAP_BPF available"
            else
                echo "FAIL: CAP_BPF not available"
                ok=false
            fi
            if capsh --print 2>/dev/null | grep -q cap_perfmon; then
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
    local script="${SCRIPT_DIR}/${name}.bt"

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pid|-p)    PID="$2"; shift 2 ;;
        --check)     check_capabilities; exit $? ;;
        --list|-l)   list_scripts; exit 0 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [SCRIPT_NAME]"
            echo ""
            echo "Options:"
            echo "  --pid PID    Attach to specific process"
            echo "  --check      Check BPF capabilities"
            echo "  --list       List available scripts"
            echo ""
            list_scripts
            exit 0
            ;;
        *)  break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "ERROR: No script specified."
    echo ""
    list_scripts
    exit 1
fi

run_script "$1"
