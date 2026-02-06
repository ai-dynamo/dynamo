#!/usr/bin/env bash
# Analyze heaptrack output
#
# Usage:
#   ./analyze_heaptrack.sh <heaptrack.gz> [output_dir]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <heaptrack.gz> [output_dir]"
    exit 1
fi

INPUT="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Handle raw heaptrack files (from LD_PRELOAD)
if [[ "$INPUT" != *.gz ]]; then
    echo "Converting raw heaptrack file..."
    PID=$(basename "$INPUT" | grep -oE '[0-9]+$' || echo "unknown")
    OUTPUT_GZ="$(dirname "$INPUT")/heaptrack.${PID}.gz"
    /usr/lib/heaptrack/libexec/heaptrack_interpret < "$INPUT" | gzip > "$OUTPUT_GZ"
    INPUT="$OUTPUT_GZ"
    echo "Created: $OUTPUT_GZ"
fi

# Output directory
if [[ $# -ge 2 ]]; then
    if [[ "$2" == */* ]]; then
        OUTPUT_DIR="$2"
    else
        OUTPUT_DIR="$SCRIPT_DIR/$2"
    fi
else
    OUTPUT_DIR="$(dirname "$INPUT")"
fi

mkdir -p "$OUTPUT_DIR"

# Copy heaptrack.gz
HEAPTRACK_GZ="$OUTPUT_DIR/heaptrack.gz"
if [[ "$(realpath "$INPUT")" != "$(realpath "$HEAPTRACK_GZ" 2>/dev/null || echo '')" ]]; then
    cp "$INPUT" "$HEAPTRACK_GZ"
fi

# Move rss.csv if exists at repo root
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "$REPO_ROOT/rss.csv" ]]; then
    mv "$REPO_ROOT/rss.csv" "$OUTPUT_DIR/rss.csv"
    echo "Moved rss.csv to output dir"
fi

echo "Generating analysis..."

# Summary
heaptrack_print "$HEAPTRACK_GZ" > "$OUTPUT_DIR/summary.txt" 2>&1 || true

# Massif format
heaptrack_print --print-massif "$OUTPUT_DIR/massif.out" "$HEAPTRACK_GZ" 2>/dev/null || true

# Flamegraph
STACKS=$(mktemp)
if heaptrack_print -F "$STACKS" "$HEAPTRACK_GZ" 2>/dev/null; then
    if [[ -x "$HOME/FlameGraph/flamegraph.pl" ]]; then
        "$HOME/FlameGraph/flamegraph.pl" < "$STACKS" > "$OUTPUT_DIR/flamegraph.svg" 2>/dev/null || true
    fi
fi
rm -f "$STACKS"

# HTML chart
if [[ -f "$OUTPUT_DIR/massif.out" ]]; then
    CMD=(python3 "$SCRIPT_DIR/massif_visualize.py" "$OUTPUT_DIR/massif.out" --html)
    if [[ -f "$OUTPUT_DIR/rss.csv" ]]; then
        CMD+=(--rss "$OUTPUT_DIR/rss.csv")
    fi
    "${CMD[@]}" > "$OUTPUT_DIR/memory_chart.html" 2>/dev/null || true
fi

echo ""
echo "Done. Output: $OUTPUT_DIR"
echo "  summary.txt"
echo "  massif.out"
echo "  flamegraph.svg"
echo "  memory_chart.html"
