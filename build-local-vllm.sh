#!/usr/bin/env bash
# Build script for Dynamo with local vLLM and/or DeepEP directories

set -e

# Default values
VLLM_PATH=""
DEEPEP_PATH=""
IMAGE_TAG="dynamo-vllm:local"
BUILD_ARGS=""
DOCKER_BUILD_ARGS=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Dynamo Docker image with local vLLM and/or DeepEP directories.

OPTIONS:
    -v, --vllm PATH         Path to local vLLM directory
    -d, --deepep PATH       Path to local DeepEP directory
    -t, --tag TAG          Docker image tag (default: dynamo-vllm:local)
    -b, --build-arg ARG    Additional Docker build arguments (can be used multiple times)
    -h, --help             Show this help message

EXAMPLES:
    # Build with local vLLM only
    $0 --vllm /path/to/vllm

    # Build with local DeepEP only
    $0 --deepep /path/to/DeepEP

    # Build with both local vLLM and DeepEP
    $0 --vllm /path/to/vllm --deepep /path/to/DeepEP

    # Build with custom tag
    $0 --vllm /path/to/vllm --tag my-custom-image:latest

    # Build with additional Docker build args
    $0 --vllm /path/to/vllm --build-arg MAX_JOBS=8

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--vllm)
            VLLM_PATH="$2"
            shift 2
            ;;
        -d|--deepep)
            DEEPEP_PATH="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -b|--build-arg)
            DOCKER_BUILD_ARGS="${DOCKER_BUILD_ARGS} --build-arg $2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if at least one local path is provided
if [ -z "$VLLM_PATH" ] && [ -z "$DEEPEP_PATH" ]; then
    print_error "At least one of --vllm or --deepep must be specified"
    usage
fi

# Validate provided paths
if [ -n "$VLLM_PATH" ] && [ ! -d "$VLLM_PATH" ]; then
    print_error "vLLM path does not exist or is not a directory: $VLLM_PATH"
    exit 1
fi

if [ -n "$DEEPEP_PATH" ] && [ ! -d "$DEEPEP_PATH" ]; then
    print_error "DeepEP path does not exist or is not a directory: $DEEPEP_PATH"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a temporary build context directory
BUILD_CONTEXT=$(mktemp -d)
print_info "Created temporary build context: $BUILD_CONTEXT"

# Cleanup function
cleanup() {
    if [ -n "$BUILD_CONTEXT" ] && [ -d "$BUILD_CONTEXT" ]; then
        print_info "Cleaning up temporary build context..."
        rm -rf "$BUILD_CONTEXT"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Copy dynamo directory contents to build context (not as subdirectory)
print_info "Copying dynamo directory to build context..."
# Use cp -a to preserve everything including hidden files
cp -a "$SCRIPT_DIR/." "$BUILD_CONTEXT/"

# Copy or link local directories to build context
if [ -n "$VLLM_PATH" ]; then
    print_info "Copying local vLLM from: $VLLM_PATH"
    cp -r "$VLLM_PATH" "$BUILD_CONTEXT/vllm"
    BUILD_ARGS="${BUILD_ARGS} --build-arg USE_LOCAL_VLLM=true"
else
    # Create empty directory to avoid COPY failure
    mkdir -p "$BUILD_CONTEXT/vllm_placeholder"
    touch "$BUILD_CONTEXT/vllm_placeholder/.placeholder"
fi

if [ -n "$DEEPEP_PATH" ]; then
    print_info "Copying local DeepEP from: $DEEPEP_PATH"
    cp -r "$DEEPEP_PATH" "$BUILD_CONTEXT/DeepEP"
    BUILD_ARGS="${BUILD_ARGS} --build-arg USE_LOCAL_DEEPEP=true"
else
    # Create empty directory to avoid COPY failure
    mkdir -p "$BUILD_CONTEXT/DeepEP_placeholder"
    touch "$BUILD_CONTEXT/DeepEP_placeholder/.placeholder"
fi

# Build the Docker image
print_info "Building Docker image with tag: $IMAGE_TAG"
print_info "Build arguments: $BUILD_ARGS $DOCKER_BUILD_ARGS"

cd "$BUILD_CONTEXT"

# Run Docker build
if DOCKER_BUILDKIT=1 docker build \
    $BUILD_ARGS \
    $DOCKER_BUILD_ARGS \
    -f container/Dockerfile.vllm \
    -t "$IMAGE_TAG" \
    . ; then
    print_info "Docker image built successfully: $IMAGE_TAG"
    
    echo
    print_info "To run the container:"
    echo "    docker run --gpus all -it $IMAGE_TAG"
    echo
    print_info "To run with volume mounts for development:"
    echo "    docker run --gpus all -it -v /path/to/workspace:/workspace $IMAGE_TAG"
else
    print_error "Docker build failed"
    exit 1
fi

print_info "Build completed successfully!"
