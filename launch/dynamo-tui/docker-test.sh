#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to run dynamo-tui tests in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if image exists
if ! docker image inspect dynamo-tui:dev >/dev/null 2>&1; then
    echo "Image not found. Building first..."
    ./launch/dynamo-tui/docker-build.sh
fi

echo "Running dynamo-tui tests in Docker..."
echo ""

# Run tests (build if needed, then test)
docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    rust:1.90.0-slim bash -c "
        apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev build-essential protobuf-compiler >/dev/null 2>&1
        cargo test -p dynamo-tui -- '$@'
    "

