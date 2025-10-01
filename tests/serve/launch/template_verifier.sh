#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(dirname "$SCRIPT_DIR")"

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $FRONTEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run ingress
python3 -m dynamo.frontend --http-port=8000 &
FRONTEND_PID=$!

# run the mock worker + template validation generate()
cd "$SCRIPT_DIR"
exec python template_verifier.py
