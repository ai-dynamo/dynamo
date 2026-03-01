#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU flame graph generation.
# Tries cargo-flamegraph, samply, or falls back to perf + flamegraph.pl.
#
# Usage:
#   ./cpu_flamegraph.sh --pid <PID>                    # attach to running process
#   ./cpu_flamegraph.sh --pid <PID> --duration 30      # 30 second capture
#   ./cpu_flamegraph.sh -- target/profiling/binary      # launch and profile

set -euo pipefail

PID=""
DURATION="${DURATION:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
OUTPUT_NAME="cpu_flamegraph_$(date +%Y%m%d_%H%M%S)"
FREQ="${FREQ:-99}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pid|-p)       PID="$2"; shift 2 ;;
        --duration|-d)  DURATION="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --output)       OUTPUT_NAME="$2"; shift 2 ;;
        --freq)         FREQ="$2"; shift 2 ;;
        --)             shift; break ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [-- <binary> [args...]]"
            echo ""
            echo "Options:"
            echo "  --pid PID         Profile running process"
            echo "  --duration N      Capture duration in seconds (default: 30)"
            echo "  --output-dir DIR  Output directory (default: .)"
            echo "  --freq HZ         Sampling frequency (default: 99)"
            exit 0
            ;;
        *) break ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Try cargo-flamegraph first (simplest)
if command -v flamegraph &>/dev/null && [[ -z "$PID" ]] && [[ $# -gt 0 ]]; then
    echo "Using cargo-flamegraph..."
    flamegraph --freq "$FREQ" --output "${OUTPUT_DIR}/${OUTPUT_NAME}.svg" -- "$@"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    exit 0
fi

# Try samply
if command -v samply &>/dev/null; then
    echo "Using samply..."
    if [[ -n "$PID" ]]; then
        samply record --pid "$PID" --duration "$DURATION" \
            --save-only --output "${OUTPUT_DIR}/${OUTPUT_NAME}.json.gz"
    else
        samply record --duration "$DURATION" \
            --save-only --output "${OUTPUT_DIR}/${OUTPUT_NAME}.json.gz" -- "$@"
    fi
    echo "Profile: ${OUTPUT_DIR}/${OUTPUT_NAME}.json.gz"
    echo "View with: samply load ${OUTPUT_DIR}/${OUTPUT_NAME}.json.gz"
    exit 0
fi

# Fallback: perf record + flamegraph.pl
if ! command -v perf &>/dev/null; then
    echo "ERROR: No profiling tool found. Install one of:"
    echo "  - cargo install flamegraph"
    echo "  - cargo install samply"
    echo "  - apt install linux-tools-\$(uname -r)"
    exit 1
fi

echo "Using perf record..."
PERF_DATA="${OUTPUT_DIR}/${OUTPUT_NAME}.perf.data"

if [[ -n "$PID" ]]; then
    perf record -F "$FREQ" -g --pid "$PID" -o "$PERF_DATA" -- sleep "$DURATION"
else
    perf record -F "$FREQ" -g -o "$PERF_DATA" -- "$@"
fi

# Generate flamegraph if flamegraph.pl is available
if command -v flamegraph.pl &>/dev/null || command -v stackcollapse-perf.pl &>/dev/null; then
    perf script -i "$PERF_DATA" | stackcollapse-perf.pl | \
        flamegraph.pl > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
elif command -v inferno-flamegraph &>/dev/null; then
    perf script -i "$PERF_DATA" | inferno-collapse-perf | \
        inferno-flamegraph > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
else
    echo "Raw perf data: $PERF_DATA"
    echo "Install flamegraph tools to generate SVG: cargo install inferno"
fi
