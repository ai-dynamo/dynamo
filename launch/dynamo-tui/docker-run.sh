#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to run dynamo-tui in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if image exists
if ! docker image inspect dynamo-tui:dev >/dev/null 2>&1; then
    echo "Image not found. Building first..."
    ./launch/dynamo-tui/docker-build.sh
fi

echo "Running dynamo-tui in Docker..."
echo "Note: Make sure ETCD and NATS are accessible from Docker"
echo ""

# Determine network mode based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use host.docker.internal
    ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://host.docker.internal:2379}"
    NATS_SERVER="${NATS_SERVER:-nats://host.docker.internal:4222}"
    NETWORK_MODE=""
else
    # Linux - use host network
    ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}"
    NATS_SERVER="${NATS_SERVER:-nats://localhost:4222}"
    NETWORK_MODE="--network host"
fi

# Run with network access to host services
docker run --rm -it \
    $NETWORK_MODE \
    -e ETCD_ENDPOINTS="$ETCD_ENDPOINTS" \
    -e NATS_SERVER="$NATS_SERVER" \
    dynamo-tui:dev \
    "$@"

