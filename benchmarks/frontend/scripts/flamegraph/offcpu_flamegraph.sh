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
STACKS_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.stacks"

# Try bpftrace-based offcputime
if command -v bpftrace &>/dev/null; then
    echo "Capturing off-CPU stacks for PID $PID for ${DURATION}s..."

    # Use timeout to limit duration
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
    ' > "$STACKS_FILE" 2>&1 || true

    echo "Stacks captured: $STACKS_FILE"

# Try bcc offcputime
elif command -v offcputime-bpfcc &>/dev/null; then
    echo "Using bcc offcputime for PID $PID for ${DURATION}s..."
    offcputime-bpfcc -d "$DURATION" -p "$PID" -m "$MIN_US" -f > "$STACKS_FILE"
else
    echo "ERROR: No BPF tool found. Install bpftrace or bcc-tools."
    exit 1
fi

# Generate flamegraph
if command -v flamegraph.pl &>/dev/null; then
    flamegraph.pl --color=io --title="Off-CPU Flame Graph (PID $PID)" \
        --countname="us" < "$STACKS_FILE" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
elif command -v inferno-flamegraph &>/dev/null; then
    inferno-flamegraph --color io --title "Off-CPU Flame Graph (PID $PID)" \
        --countname "us" < "$STACKS_FILE" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
else
    echo "Raw stacks: $STACKS_FILE"
    echo "Install flamegraph tools to generate SVG: cargo install inferno"
fi
