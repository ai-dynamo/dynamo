#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs S3 integration tests against a local MinIO instance.
# Usage: bash lib/kvbm-engine/scripts/test-s3.sh

set -euo pipefail

MINIO_CONTAINER="kvbm-minio-test-$$-$RANDOM"
MINIO_PORT="${MINIO_PORT:-9876}"
MINIO_ROOT_USER="minioadmin"
MINIO_ROOT_PASSWORD="minioadmin"

cleanup() {
    echo "Cleaning up MinIO container: $MINIO_CONTAINER"
    docker stop "$MINIO_CONTAINER" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting MinIO container: $MINIO_CONTAINER on port $MINIO_PORT"
docker run --rm -d \
    --name "$MINIO_CONTAINER" \
    -p "${MINIO_PORT}:9000" \
    -e "MINIO_ROOT_USER=${MINIO_ROOT_USER}" \
    -e "MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}" \
    minio/minio:latest server /data

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${MINIO_PORT}/minio/health/live" >/dev/null 2>&1; then
        echo "MinIO is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: MinIO failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

export S3_TEST_ENDPOINT="http://localhost:${MINIO_PORT}"
export AWS_ACCESS_KEY_ID="${MINIO_ROOT_USER}"
export AWS_SECRET_ACCESS_KEY="${MINIO_ROOT_PASSWORD}"
export AWS_DEFAULT_REGION="us-east-1"

echo "Running S3 integration tests..."
timeout 120 cargo test -p kvbm-engine --features testing-s3 -- s3_integration
exit_code=$?

echo "Tests finished with exit code: $exit_code"
exit $exit_code
