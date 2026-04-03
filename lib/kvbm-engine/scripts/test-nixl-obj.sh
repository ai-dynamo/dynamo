#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs NIXL OBJ integration tests (DRAM→S3→DRAM round-trip) inside Docker.
#
# What this does:
#   1. Builds the kvbm-nixl-obj-test Docker image (NIXL 0.10.0 + UCX + OBJ backend).
#   2. Starts MinIO and the test runner via Docker Compose.
#   3. Exits with the test runner's exit code.
#
# The build compiles UCX, AWS SDK C++, and NIXL from source — expect 15-30 min
# on first run. Subsequent runs use Docker layer cache and are much faster.
#
# Usage:
#   bash lib/kvbm-engine/scripts/test-nixl-obj.sh [--no-cache]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.nixl-obj-test.yml"
DOCKERFILE="${REPO_ROOT}/lib/kvbm-engine/container/Dockerfile.nixl-obj-test"
IMAGE_NAME="kvbm-nixl-obj-test:latest"

NO_CACHE=""
if [[ "${1:-}" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
fi

cleanup() {
    echo "Stopping Docker Compose services..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Building NIXL OBJ test image (this may take 15-30 min on first run) ==="
docker build ${NO_CACHE} \
    -f "$DOCKERFILE" \
    -t "$IMAGE_NAME" \
    "$REPO_ROOT"

echo ""
echo "=== Starting MinIO + NIXL test runner ==="
docker compose -f "$COMPOSE_FILE" up \
    --abort-on-container-exit \
    --exit-code-from nixl-test-runner

exit_code=$?
echo ""
echo "Tests finished with exit code: ${exit_code}"
exit ${exit_code}
