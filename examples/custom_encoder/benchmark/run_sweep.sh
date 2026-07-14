#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD_DIR="${WORKLOAD_DIR:-$SCRIPT_DIR/.data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)}"

python "$SCRIPT_DIR/run_image_sweep.py" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

python "$SCRIPT_DIR/validate_results.py" "$OUTPUT_DIR"
python "$SCRIPT_DIR/summarize_results.py" "$OUTPUT_DIR" \
    --markdown "$OUTPUT_DIR/benchmark.md" \
    --csv "$OUTPUT_DIR/benchmark.csv"

echo "benchmark=$OUTPUT_DIR/benchmark.md"
