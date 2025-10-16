#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to build vLLM container with runtime and tag/push to NVCR
#
# Usage:
#   ./scripts/build-push-vllm-runtime.sh [OPTIONS]
#
# Examples:
#   # Build and push with default settings
#   ./scripts/build-push-vllm-runtime.sh
#
#   # Build only (no push)
#   ./scripts/build-push-vllm-runtime.sh --no-push
#
#   # Dry run to see what commands would be executed
#   ./scripts/build-push-vllm-runtime.sh --dry-run
#
#   # Custom registry and namespace
#   ./scripts/build-push-vllm-runtime.sh --registry nvcr.io/custom --namespace my-namespace
#
#   # Use specific version tag
#   ./scripts/build-push-vllm-runtime.sh --version v1.2.3
#

set -e

# Default configuration
DEFAULT_REGISTRY="nvcr.io/nvidian/dynamo-dev"
DEFAULT_NAMESPACE="biswa"
DEFAULT_FRAMEWORK="vllm"
DEFAULT_TARGET="runtime"
BUILD_SCRIPT_PATH="$(dirname "$(readlink -f "$0")")/../container/build.sh"

# Configuration variables
REGISTRY="$DEFAULT_REGISTRY"
NAMESPACE="$DEFAULT_NAMESPACE"
FRAMEWORK="$DEFAULT_FRAMEWORK"
TARGET="$DEFAULT_TARGET"
VERSION=""
NO_PUSH=false
DRY_RUN=false
NO_CACHE=false
BUILD_ARGS=""
PLATFORM="linux/amd64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build vLLM container with runtime and tag/push to NVCR.

OPTIONS:
    -h, --help              Show this help message
    --registry REGISTRY     Container registry (default: $DEFAULT_REGISTRY)
    --namespace NAMESPACE   Registry namespace (default: $DEFAULT_NAMESPACE)
    --framework FRAMEWORK   Framework to build (default: $DEFAULT_FRAMEWORK)
    --target TARGET         Build target (default: $DEFAULT_TARGET)
    --version VERSION       Version tag (auto-detected from git if not provided)
    --platform PLATFORM     Platform for docker build (default: $PLATFORM)
    --no-push              Build only, don't push to registry
    --no-cache             Disable docker build cache
    --dry-run              Print commands without executing them
    --build-arg ARG        Additional build arguments to pass to build.sh

EXAMPLES:
    # Build and push with default settings
    $(basename "$0")

    # Build only (no push)
    $(basename "$0") --no-push

    # Dry run to see what commands would be executed
    $(basename "$0") --dry-run

    # Custom registry and namespace
    $(basename "$0") --registry nvcr.io/custom --namespace my-namespace

    # Use specific version tag
    $(basename "$0") --version v1.2.3

    # Build with additional arguments
    $(basename "$0") --build-arg "--use-sccache" --build-arg "--sccache-bucket my-bucket"

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY        Override default registry
    DOCKER_NAMESPACE       Override default namespace
    NO_DOCKER_PUSH         Set to 'true' to skip pushing (same as --no-push)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-push)
            NO_PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS $2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Override with environment variables if set
if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
    REGISTRY="$DOCKER_REGISTRY"
fi

if [[ -n "${DOCKER_NAMESPACE:-}" ]]; then
    NAMESPACE="$DOCKER_NAMESPACE"
fi

if [[ "${NO_DOCKER_PUSH:-}" == "true" ]]; then
    NO_PUSH=true
fi

# Validate build script exists
if [[ ! -f "$BUILD_SCRIPT_PATH" ]]; then
    log_error "Build script not found at: $BUILD_SCRIPT_PATH"
    exit 1
fi

# Auto-detect version if not provided
if [[ -z "$VERSION" ]]; then
    # Get short commit hash
    commit_id=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    # Check if current commit matches a tag
    current_tag=$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//') || true

    # Get latest tag and add commit ID for dev builds
    latest_tag=$(git describe --tags --abbrev=0 "$(git rev-list --tags --max-count=1 main 2>/dev/null)" 2>/dev/null | sed 's/^v//') || true
    if [[ -z ${latest_tag} ]]; then
        latest_tag="0.1.0"
        log_warning "No git release tag found, using default version: ${latest_tag}"
    fi

    # Use tag if available, otherwise use latest_tag.dev.commit_id
    VERSION="v${current_tag:-$latest_tag.dev.$commit_id}"
fi

# Construct image names
LOCAL_IMAGE="dynamo:${VERSION}-${FRAMEWORK}-${TARGET}"
REMOTE_IMAGE="${REGISTRY}/${NAMESPACE}:${VERSION}-${FRAMEWORK}-${TARGET}"

# Show configuration
log_info "Build Configuration:"
echo "  Registry: $REGISTRY"
echo "  Namespace: $NAMESPACE"
echo "  Framework: $FRAMEWORK"
echo "  Target: $TARGET"
echo "  Version: $VERSION"
echo "  Platform: $PLATFORM"
echo "  Local Image: $LOCAL_IMAGE"
echo "  Remote Image: $REMOTE_IMAGE"
echo "  Build Script: $BUILD_SCRIPT_PATH"
echo "  No Push: $NO_PUSH"
echo "  Dry Run: $DRY_RUN"
if [[ -n "$BUILD_ARGS" ]]; then
    echo "  Build Args: $BUILD_ARGS"
fi
echo ""

# Prepare build command
BUILD_CMD="$BUILD_SCRIPT_PATH"
BUILD_CMD="$BUILD_CMD --framework $FRAMEWORK"
BUILD_CMD="$BUILD_CMD --target $TARGET"
BUILD_CMD="$BUILD_CMD --platform $PLATFORM"
BUILD_CMD="$BUILD_CMD --tag $LOCAL_IMAGE"

if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    BUILD_CMD="$BUILD_CMD --dry-run"
fi

# Add additional build arguments
if [[ -n "$BUILD_ARGS" ]]; then
    BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
fi

# Execute build
log_info "Building container image..."
echo "Command: $BUILD_CMD"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute: $BUILD_CMD"
else
    if ! eval "$BUILD_CMD"; then
        log_error "Build failed!"
        exit 1
    fi
    log_success "Build completed successfully!"
fi

# Tag for remote registry
log_info "Tagging image for registry..."
TAG_CMD="docker tag $LOCAL_IMAGE $REMOTE_IMAGE"
echo "Command: $TAG_CMD"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute: $TAG_CMD"
else
    if ! eval "$TAG_CMD"; then
        log_error "Tagging failed!"
        exit 1
    fi
    log_success "Image tagged successfully!"
fi

# Push to registry (unless --no-push is specified)
if [[ "$NO_PUSH" == "true" ]]; then
    log_info "Skipping push (--no-push specified)"
    log_success "Build and tag completed. Image ready: $REMOTE_IMAGE"
else
    log_info "Pushing image to registry..."
    PUSH_CMD="docker push $REMOTE_IMAGE"
    echo "Command: $PUSH_CMD"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute: $PUSH_CMD"
    else
        if ! eval "$PUSH_CMD"; then
            log_error "Push failed!"
            log_error "Make sure you're logged in to the registry:"
            log_error "  docker login $REGISTRY"
            exit 1
        fi
        log_success "Image pushed successfully!"
    fi
fi

# Show final summary
echo ""
log_success "=== SUMMARY ==="
echo "Local Image:  $LOCAL_IMAGE"
echo "Remote Image: $REMOTE_IMAGE"
if [[ "$NO_PUSH" == "false" && "$DRY_RUN" == "false" ]]; then
    echo ""
    log_info "Image is now available at: $REMOTE_IMAGE"
    log_info "To pull and run:"
    echo "  docker pull $REMOTE_IMAGE"
    echo "  docker run --rm -it --gpus all $REMOTE_IMAGE"
elif [[ "$NO_PUSH" == "true" && "$DRY_RUN" == "false" ]]; then
    echo ""
    log_info "To push manually:"
    echo "  docker push $REMOTE_IMAGE"
elif [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    log_info "This was a dry run. No actual operations were performed."
fi
