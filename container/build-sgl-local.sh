#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for Dockerfile.sglang-local
# Automatically reads AWS credentials and endpoint from environment variables
#
# Usage:
#   ./build-sgl-local.sh [OPTIONS]
#
# Environment variables used:
#   - AWS_ACCESS_KEY_ID (required for S3 cache)
#   - AWS_SECRET_ACCESS_KEY (required for S3 cache)
#   - AWS_ENDPOINT_URL (optional, for custom S3 endpoints like LocalStack)
#   - SCCACHE_BUCKET (optional, defaults to "sccache")
#   - SCCACHE_REGION (optional, defaults to "us-east-1")
#   - SGLANG_IMAGE_TAG (optional, defaults to "v0.5.6.post2-cu130-runtime")
#   - CUDA_VERSION (optional, defaults to "13")
#   - BRANCH_TYPE (optional: "local", "remote", or unset for PyPI install)
#   - DYNAMO_VERSION (optional, PyPI version when BRANCH_TYPE is not set)
#   - CARGO_BUILD_JOBS (optional, defaults to 16)

set -euo pipefail

# Get script directory and set build context to parent (dynamo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONTEXT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUILD_CONTEXT}"

# Default values
SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG:-v0.5.6.post2-cu130-runtime}"
CUDA_VERSION="${CUDA_VERSION:-13}"
SCCACHE_BUCKET="${SCCACHE_BUCKET:-sccache}"
SCCACHE_REGION="${SCCACHE_REGION:-us-east-1}"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"

# Parse command line arguments
TAG=""
TARGET=""
NO_CACHE=""
PLATFORM=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag TAG              Docker image tag (required)"
            echo "  --target TARGET         Build target/stage (optional)"
            echo "  --no-cache             Disable build cache"
            echo "  --platform PLATFORM    Target platform (e.g., linux/amd64)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  AWS_ACCESS_KEY_ID      AWS access key (required for S3 cache)"
            echo "  AWS_SECRET_ACCESS_KEY   AWS secret key (required for S3 cache)"
            echo "  AWS_ENDPOINT_URL        Custom S3 endpoint (optional)"
            echo "  SCCACHE_BUCKET          S3 bucket for cache (default: sccache)"
            echo "  SCCACHE_REGION          AWS region (default: us-east-1)"
            echo "  SGLANG_IMAGE_TAG        SGLang base image tag"
            echo "  CUDA_VERSION            CUDA version (12 or 13, default: 13)"
            echo "  BRANCH_TYPE             Build type: local, remote, or unset for PyPI"
            echo "  DYNAMO_VERSION          PyPI version when BRANCH_TYPE not set"
            echo "  CARGO_BUILD_JOBS        Number of parallel cargo jobs (default: 16)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TAG" ]]; then
    echo "Error: --tag is required"
    echo "Run with --help for usage information"
    exit 1
fi

# Check for AWS credentials (warn if missing, but don't fail - user might not need S3 cache)
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "Warning: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set"
    echo "S3 cache will not be available. Build will still proceed."
fi

# Build arguments
BUILD_ARGS=(
    --build-arg "SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG}"
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "SCCACHE_BUCKET=${SCCACHE_BUCKET}"
    --build-arg "SCCACHE_REGION=${SCCACHE_REGION}"
    --build-arg "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}"
)

# Add optional build args if set
if [[ -n "${BRANCH_TYPE:-}" ]]; then
    BUILD_ARGS+=(--build-arg "BRANCH_TYPE=${BRANCH_TYPE}")
fi

if [[ -n "${DYNAMO_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "DYNAMO_VERSION=${DYNAMO_VERSION}")
fi

# Add AWS_ENDPOINT_URL if set
if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
    BUILD_ARGS+=(--build-arg "AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}")
fi

# Add secrets for AWS credentials (using env type to read from environment)
SECRET_ARGS=()
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    SECRET_ARGS+=(--secret "id=AWS_ACCESS_KEY_ID,env=AWS_ACCESS_KEY_ID")
fi
if [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    SECRET_ARGS+=(--secret "id=AWS_SECRET_ACCESS_KEY,env=AWS_SECRET_ACCESS_KEY")
fi

# Build target string
TARGET_STR=""
if [[ -n "$TARGET" ]]; then
    TARGET_STR="--target $TARGET"
fi

# Enable BuildKit for secret support
export DOCKER_BUILDKIT=1

# Build the image
echo "Building Docker image: ${TAG}"
echo "  SGLang image: ${SGLANG_IMAGE_TAG}"
echo "  CUDA version: ${CUDA_VERSION}"
echo "  SCCACHE bucket: ${SCCACHE_BUCKET}"
echo "  SCCACHE region: ${SCCACHE_REGION}"
if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
    echo "  AWS endpoint: ${AWS_ENDPOINT_URL}"
fi
if [[ -n "${BRANCH_TYPE:-}" ]]; then
    echo "  Branch type: ${BRANCH_TYPE}"
fi
echo ""

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.sglang-local" \
    ${TARGET_STR} \
    ${PLATFORM} \
    "${BUILD_ARGS[@]}" \
    "${SECRET_ARGS[@]}" \
    ${NO_CACHE} \
    --tag "${TAG}" \
    "${EXTRA_ARGS[@]}" \
    .

echo ""
echo "Build complete! Image tagged as: ${TAG}"

