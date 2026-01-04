#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to build and test dynamo-tui in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Building dynamo-tui in Docker..."
docker build \
    -f launch/dynamo-tui/Dockerfile.dev \
    -t dynamo-tui:dev \
    .

echo ""
echo "Build complete! You can now run:"
echo "  ./launch/dynamo-tui/docker-run.sh"
echo ""
echo "Or run tests with:"
echo "  ./launch/dynamo-tui/docker-test.sh"

