#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
OSLS="${OSLS:-1 70}"
REQUEST_COUNT="${REQUEST_COUNT:-100}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-$REPO_ROOT/logs/qwen3_vl_custom_encoder/sweep_$RUN_ID}"

# Optional path to the append-only topology/server log. When supplied, each
# aiperf result directory receives only the server lines emitted during that
# run, allowing summarize_results.py to join stage timings with aiperf metrics.
SERVER_LOG="${SERVER_LOG:-}"

if [[ -n "$SERVER_LOG" && ! -f "$SERVER_LOG" ]]; then
    echo "SERVER_LOG does not exist: $SERVER_LOG" >&2
    exit 1
fi

mkdir -p "$SWEEP_DIR"
echo "sweep_dir=$SWEEP_DIR"

for osl in $OSLS; do
    for concurrency in $CONCURRENCIES; do
        artifact_dir="$SWEEP_DIR/osl${osl}/conc${concurrency}"
        if [[ -e "$artifact_dir/profile_export_aiperf.json" ]]; then
            echo "Refusing to overwrite completed run: $artifact_dir" >&2
            exit 1
        fi

        server_log_start=0
        if [[ -n "$SERVER_LOG" ]]; then
            server_log_start="$(wc -l < "$SERVER_LOG")"
        fi

        echo "run osl=$osl concurrency=$concurrency requests=$REQUEST_COUNT"
        OSL="$osl" \
        CONCURRENCY="$concurrency" \
        REQUEST_COUNT="$REQUEST_COUNT" \
        WARMUP_REQUEST_COUNT="$WARMUP_REQUEST_COUNT" \
        ARTIFACT_DIR="$artifact_dir" \
            "$SCRIPT_DIR/run_aiperf.sh"

        if [[ -n "$SERVER_LOG" ]]; then
            server_log_end="$(wc -l < "$SERVER_LOG")"
            if ((server_log_end > server_log_start)); then
                sed -n "$((server_log_start + 1)),${server_log_end}p" \
                    "$SERVER_LOG" > "$artifact_dir/custom_encoder.log"
            else
                : > "$artifact_dir/custom_encoder.log"
            fi
        fi
    done
done

python "$SCRIPT_DIR/summarize_results.py" \
    "$SWEEP_DIR" \
    --output-dir "$SWEEP_DIR"
