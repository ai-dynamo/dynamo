#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for Dynamo with local/pre-built vLLM images

set -e

# Default values
LOCAL_VLLM_IMAGE="vllm-elastic-ep:latest_all2all_buffer_input"
DYNAMO_BASE_TAG="dynamo:latest-none"
OUTPUT_TAG="my-dynamo-vllm:local"
TARGET="dev"
NO_CACHE=""
BUILD_DYNAMO_BASE=true
DOCKERFILE_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname $(dirname $(dirname $(dirname "$DOCKERFILE_DIR"))))

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build Dynamo image using a local/pre-built vLLM image"
    echo ""
    echo "Options:"
    echo "  --vllm-image IMAGE        Local vLLM image to use (default: $LOCAL_VLLM_IMAGE)"
    echo "  --tag TAG                 Output image tag (default: $OUTPUT_TAG)"
    echo "  --target TARGET           Build target: runtime or dev (default: $TARGET)"
    echo "  --no-cache               Disable Docker build cache"
    echo "  --skip-base              Skip building dynamo base (assumes it exists)"
    echo "  --dynamo-base TAG        Dynamo base image tag (default: $DYNAMO_BASE_TAG)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Use default vLLM image"
    echo "  $0"
    echo ""
    echo "  # Use custom vLLM image"
    echo "  $0 --vllm-image my-vllm:custom --tag my-dynamo:test"
    echo ""
    echo "  # Build runtime image only"
    echo "  $0 --target runtime --tag my-dynamo:prod"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-image)
            LOCAL_VLLM_IMAGE="$2"
            shift 2
            ;;
        --tag)
            OUTPUT_TAG="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            if [[ "$TARGET" != "runtime" && "$TARGET" != "dev" ]]; then
                echo -e "${RED}Error: --target must be 'runtime' or 'dev'${NC}"
                exit 1
            fi
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --skip-base)
            BUILD_DYNAMO_BASE=false
            shift
            ;;
        --dynamo-base)
            DYNAMO_BASE_TAG="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Dynamo with Local vLLM Image${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  vLLM Image: $LOCAL_VLLM_IMAGE"
echo "  Output Tag: $OUTPUT_TAG"
echo "  Target: $TARGET"
echo "  Dynamo Base: $DYNAMO_BASE_TAG"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Check if local vLLM image exists
if ! docker image inspect "$LOCAL_VLLM_IMAGE" > /dev/null 2>&1; then
    echo -e "${RED}Error: Local vLLM image '$LOCAL_VLLM_IMAGE' not found${NC}"
    echo "Available vLLM images:"
    docker images | grep -E "^REPOSITORY|vllm" || echo "No vLLM images found"
    exit 1
fi

# Step 1: Build Dynamo base if requested
if [ "$BUILD_DYNAMO_BASE" = true ]; then
    echo -e "${YELLOW}Step 1: Building Dynamo base image...${NC}"
    cd "$PROJECT_ROOT"

    # Check if build.sh exists
    if [ ! -f "container/build.sh" ]; then
        echo -e "${RED}Error: container/build.sh not found in $PROJECT_ROOT${NC}"
        exit 1
    fi

    ./container/build.sh \
        --framework none \
        --tag "$DYNAMO_BASE_TAG" \
        $NO_CACHE

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to build Dynamo base image${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dynamo base image built successfully${NC}"
else
    echo -e "${YELLOW}Step 1: Skipping Dynamo base build (using existing)${NC}"
    # Check if base image exists
    if ! docker image inspect "$DYNAMO_BASE_TAG" > /dev/null 2>&1; then
        echo -e "${RED}Error: Dynamo base image '$DYNAMO_BASE_TAG' not found${NC}"
        echo "Please build it first or remove --skip-base flag"
        exit 1
    fi
fi

# Step 2: Build combined image with local vLLM
echo ""
echo -e "${YELLOW}Step 2: Building combined Dynamo + vLLM image...${NC}"
cd "$PROJECT_ROOT"

# Build the combined image
docker build \
    -f "$DOCKERFILE_DIR/Dockerfile.local_vllm" \
    --build-arg LOCAL_VLLM_IMAGE="$LOCAL_VLLM_IMAGE" \
    --build-arg DYNAMO_BASE_IMAGE="$DYNAMO_BASE_TAG" \
    --target "$TARGET" \
    --tag "$OUTPUT_TAG" \
    $NO_CACHE \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to build combined image${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output image: $OUTPUT_TAG"
echo ""
echo "To test the image:"
echo "  docker run --rm -it --gpus all $OUTPUT_TAG python -c 'import vllm; print(vllm.__version__)'"
echo ""
echo "To use in pytest:"
echo "  pytest tests/fault_tolerance/deploy/test_deployment.py::test_fault_scenario[vllm-moe-agg-tp-1-dp-2-none] \\"
echo "    --image $OUTPUT_TAG \\"
echo "    --namespace dynamo-kubernetes \\"
echo "    -v -s"
echo ""
echo "To push to registry:"
echo "  docker tag $OUTPUT_TAG <your-registry>/$OUTPUT_TAG"
echo "  docker push <your-registry>/$OUTPUT_TAG"

