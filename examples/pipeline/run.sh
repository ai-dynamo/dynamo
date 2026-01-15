#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Simple pipeline demo runner
# Starts all stages and runs the client

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Track PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT INT TERM

echo "=== Simple Pipeline Demo ==="
echo ""

# Start Stage 3 (backend)
echo "Starting Stage 3..."
python3 stage3.py &
PIDS+=($!)
sleep 1

# Start Stage 2 (middle)
echo "Starting Stage 2..."
python3 stage2.py &
PIDS+=($!)
sleep 1

# Start Stage 1 (entry)
echo "Starting Stage 1..."
python3 stage1.py &
PIDS+=($!)
sleep 1

wait
