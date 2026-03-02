#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Off-CPU flame graph using BPF offcputime + flamegraph.
# Shows what threads are blocked on (mutexes, I/O, futex, socket waits).
#
# Usage:
#   sudo ./offcpu_flamegraph.sh --pid <PID>
#   sudo ./offcpu_flamegraph.sh --pid <PID> --duration 30

set -euo pipefail

PID=""
DURATION="${DURATION:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
OUTPUT_NAME="offcpu_flamegraph_$(date +%Y%m%d_%H%M%S)"
MIN_US="${MIN_US:-1000}"  # Minimum off-CPU time to record (1ms)

while [[ $# -gt 0 ]]; do
    case $1 in
        --pid|-p)       PID="$2"; shift 2 ;;
        --duration|-d)  DURATION="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --output)       OUTPUT_NAME="$2"; shift 2 ;;
        --min-us)       MIN_US="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sudo $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pid PID         Target process (required)"
            echo "  --duration N      Capture duration in seconds (default: 30)"
            echo "  --output-dir DIR  Output directory (default: .)"
            echo "  --min-us N        Minimum off-CPU microseconds to record (default: 1000)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PID" ]]; then
    echo "ERROR: --pid is required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
RAW_STACKS="${OUTPUT_DIR}/${OUTPUT_NAME}.raw"
FOLDED_STACKS="${OUTPUT_DIR}/${OUTPUT_NAME}.stacks"
NEEDS_FOLD=false

# Try bpftrace-based offcputime
if command -v bpftrace &>/dev/null; then
    echo "Capturing off-CPU stacks for PID $PID for ${DURATION}s..."

    # Use timeout to limit duration; keep stderr separate from stack data
    timeout "$DURATION" bpftrace -p "$PID" -e '
        tracepoint:sched:sched_switch {
            if (args.prev_state != 0) {
                @off[tid] = nsecs;
                @stack[tid] = kstack;
            }
        }
        tracepoint:sched:sched_switch {
            $start = @off[args.next_pid];
            if ($start) {
                $delta = (nsecs - $start) / 1000;
                if ($delta > '"$MIN_US"') {
                    @stacks[@stack[args.next_pid], comm] = sum($delta);
                }
                delete(@off[args.next_pid]);
                delete(@stack[args.next_pid]);
            }
        }
        END { print(@stacks); clear(@off); clear(@stack); }
    ' > "$RAW_STACKS" 2>/dev/null || true

    NEEDS_FOLD=true
    echo "Raw stacks captured: $RAW_STACKS"

# Try bcc offcputime — outputs folded format directly with -f
elif command -v offcputime-bpfcc &>/dev/null; then
    echo "Using bcc offcputime for PID $PID for ${DURATION}s..."
    offcputime-bpfcc -d "$DURATION" -p "$PID" -m "$MIN_US" -f > "$FOLDED_STACKS"
else
    echo "ERROR: No BPF tool found. Install bpftrace or bcc-tools."
    exit 1
fi

# Convert bpftrace native format to folded stacks.
# bpftrace @stacks[kstack, comm] format:
#   @stacks[
#       leaf_func+offset
#       ...
#       root_func+offset
#   , comm_name]: value
# Folded format: comm;root_func;...;leaf_func value
if [[ "$NEEDS_FOLD" == true ]] && [[ -f "$RAW_STACKS" ]]; then
    awk '
    /^@stacks\[/ { n=0; next }
    /^[[:space:]]+[a-zA-Z_]/ {
        gsub(/^[[:space:]]+/, "")
        sub(/\+[0-9]+$/, "")
        frames[n++] = $0
        next
    }
    /^, / {
        sub(/^, /, "")
        idx = index($0, "]: ")
        comm = substr($0, 1, idx-1)
        val = substr($0, idx+3) + 0
        if (n > 0 && val > 0) {
            printf "%s", comm
            for (i=n-1; i>=0; i--) printf ";%s", frames[i]
            printf " %d\n", val
        }
        n = 0
        next
    }
    ' "$RAW_STACKS" > "$FOLDED_STACKS"
    echo "Folded stacks: $FOLDED_STACKS ($(wc -l < "$FOLDED_STACKS") entries)"
fi

# Generate flamegraph SVG from folded stacks
if [[ ! -s "$FOLDED_STACKS" ]]; then
    echo "WARNING: No stacks captured — SVG not generated"
    exit 0
fi

if command -v flamegraph.pl &>/dev/null; then
    flamegraph.pl --color=io --title="Off-CPU Flame Graph (PID $PID)" \
        --countname="us" < "$FOLDED_STACKS" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
elif command -v inferno-flamegraph &>/dev/null; then
    inferno-flamegraph --colors io --title "Off-CPU Flame Graph (PID $PID)" \
        --countname "us" < "$FOLDED_STACKS" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
else
    echo "Folded stacks: $FOLDED_STACKS"
    echo "Install flamegraph tools to generate SVG: cargo install inferno"
fi
