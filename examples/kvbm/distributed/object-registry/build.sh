#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set -e

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Defaults
TAG="registry-hub:latest"
PLATFORM="linux/amd64"
ARCH_ALT="x86_64"
NO_CACHE=""
DRY_RUN=""

# NIXL configuration
NIXL_REF="0.7.1"
NIXL_UCX_REF="v1.19.0"

# sccache configuration
USE_SCCACHE=""
SCCACHE_BUCKET=""
SCCACHE_REGION=""
SCCACHE_S3_ENDPOINT=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build the Registry Hub Docker image.

Options:
  -h, --help              Show this help message
  --tag TAG               Image tag (default: registry-hub:latest)
  --platform PLATFORM     Build platform (default: linux/amd64)
  --no-cache              Disable Docker build cache
  --dry-run               Print docker commands without running

sccache options (for faster rebuilds):
  --use-sccache           Enable sccache for Rust compilation caching
  --sccache-bucket        S3 bucket name for sccache (required with --use-sccache)
  --sccache-region        S3 region for sccache (required with --use-sccache)
  --sccache-endpoint      S3 endpoint URL (for MinIO/custom S3)

Examples:
  # Basic build
  $(basename "$0")

  # Build with custom tag
  $(basename "$0") --tag my-registry:v1.0

  # Build with sccache
  export AWS_ACCESS_KEY_ID=your_key
  export AWS_SECRET_ACCESS_KEY=your_secret
  $(basename "$0") --use-sccache --sccache-bucket mybucket --sccache-region us-east-1
EOF
    exit 0
}

error() {
    echo "ERROR: $1" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            if [[ "$PLATFORM" == *"arm64"* ]]; then
                ARCH_ALT="aarch64"
            fi
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --dry-run)
            DRY_RUN="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            shift
            ;;
        --use-sccache)
            USE_SCCACHE="true"
            shift
            ;;
        --sccache-bucket)
            SCCACHE_BUCKET="$2"
            shift 2
            ;;
        --sccache-region)
            SCCACHE_REGION="$2"
            shift 2
            ;;
        --sccache-endpoint)
            SCCACHE_S3_ENDPOINT="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate sccache configuration
if [ "$USE_SCCACHE" = "true" ]; then
    if [ -z "$SCCACHE_BUCKET" ]; then
        error "--sccache-bucket is required when --use-sccache is specified"
    fi
    if [ -z "$SCCACHE_REGION" ]; then
        error "--sccache-region is required when --use-sccache is specified"
    fi
fi

# Build arguments
BUILD_ARGS="--build-arg ARCH_ALT=${ARCH_ALT}"
BUILD_ARGS+=" --build-arg NIXL_REF=${NIXL_REF}"
BUILD_ARGS+=" --build-arg NIXL_UCX_REF=${NIXL_UCX_REF}"

if [ "$USE_SCCACHE" = "true" ]; then
    BUILD_ARGS+=" --build-arg USE_SCCACHE=true"
    BUILD_ARGS+=" --build-arg SCCACHE_BUCKET=${SCCACHE_BUCKET}"
    BUILD_ARGS+=" --build-arg SCCACHE_REGION=${SCCACHE_REGION}"
    if [ -n "$SCCACHE_S3_ENDPOINT" ]; then
        BUILD_ARGS+=" --build-arg SCCACHE_S3_ENDPOINT=${SCCACHE_S3_ENDPOINT}"
    fi
    BUILD_ARGS+=" --secret id=aws-key-id,env=AWS_ACCESS_KEY_ID"
    BUILD_ARGS+=" --secret id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY"
fi

echo ""
echo "======================================"
echo "Building Registry Hub Image"
echo "======================================"
echo "  Tag: ${TAG}"
echo "  Platform: ${PLATFORM}"
echo "  NIXL: ${NIXL_REF}"
echo "  UCX: ${NIXL_UCX_REF}"
if [ "$USE_SCCACHE" = "true" ]; then
    echo "  sccache: Enabled"
    echo "  sccache Bucket: ${SCCACHE_BUCKET}"
    echo "  sccache Region: ${SCCACHE_REGION}"
    [ -n "$SCCACHE_S3_ENDPOINT" ] && echo "  sccache Endpoint: ${SCCACHE_S3_ENDPOINT}"
fi
echo ""

# Run the build
$DRY_RUN docker build \
    --platform "${PLATFORM}" \
    --file "${SOURCE_DIR}/Dockerfile.registry-hub" \
    --tag "${TAG}" \
    ${BUILD_ARGS} \
    ${NO_CACHE} \
    "${BUILD_CONTEXT}"

echo ""
echo "======================================"
echo "Build Complete!"
echo "======================================"
echo ""
echo "Run with:"
echo "  docker run -p 5555:5555 -p 5556:5556 ${TAG}"
echo ""

