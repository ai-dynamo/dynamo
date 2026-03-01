#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Differential flame graph (before/after comparison).
# Shows what changed between two profiles — red = regression, blue = improvement.
#
# Usage:
#   ./diff_flamegraph.sh <before.perf.data> <after.perf.data>
#   ./diff_flamegraph.sh <before.stacks> <after.stacks>

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-.}"
OUTPUT_NAME="diff_flamegraph_$(date +%Y%m%d_%H%M%S)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <before> <after>"
    echo ""
    echo "Accepts .perf.data files or pre-folded .stacks files."
    echo "Output: differential SVG flamegraph (red=regression, blue=improvement)"
    exit 1
fi

BEFORE="$1"
AFTER="$2"

while [[ $# -gt 2 ]] && [[ $# -gt 0 ]]; do
    shift 2
    case ${1:-} in
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --output)      OUTPUT_NAME="$2"; shift 2 ;;
        *)  break ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Convert perf.data to folded stacks if needed
fold_perf_data() {
    local input=$1
    local output=$2

    if [[ "$input" == *.perf.data ]]; then
        if command -v stackcollapse-perf.pl &>/dev/null; then
            perf script -i "$input" | stackcollapse-perf.pl > "$output"
        elif command -v inferno-collapse-perf &>/dev/null; then
            perf script -i "$input" | inferno-collapse-perf > "$output"
        else
            echo "ERROR: Need stackcollapse-perf.pl or inferno-collapse-perf"
            exit 1
        fi
    else
        cp "$input" "$output"
    fi
}

BEFORE_FOLDED=$(mktemp)
AFTER_FOLDED=$(mktemp)
trap 'rm -f "$BEFORE_FOLDED" "$AFTER_FOLDED"' EXIT

fold_perf_data "$BEFORE" "$BEFORE_FOLDED"
fold_perf_data "$AFTER" "$AFTER_FOLDED"

if command -v difffolded.pl &>/dev/null && command -v flamegraph.pl &>/dev/null; then
    difffolded.pl "$BEFORE_FOLDED" "$AFTER_FOLDED" | \
        flamegraph.pl --title="Differential Flame Graph" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Diff flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
elif command -v inferno-diff-folded &>/dev/null && command -v inferno-flamegraph &>/dev/null; then
    inferno-diff-folded "$BEFORE_FOLDED" "$AFTER_FOLDED" | \
        inferno-flamegraph --title "Differential Flame Graph" > "${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
    echo "Diff flame graph: ${OUTPUT_DIR}/${OUTPUT_NAME}.svg"
else
    echo "ERROR: Need flamegraph tools (Brendan Gregg's or inferno)"
    echo "  cargo install inferno"
    echo "  or: git clone https://github.com/brendangregg/FlameGraph"
    exit 1
fi
