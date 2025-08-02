#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script takes an existing pre-blt image from CI and turns it into a dev image.
# It will be tagged as the same image but with -dev suffix.
#
# Usage: ./build-dev-image.sh [OPTIONS] <ci-image-name>
# Example: ./build-dev-image.sh gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:66231cf0977716a60dc082c344f7e81a245929f3-32632154-vllm-amd64
# Result: Creates dynamo:latest-vllm-dev

set -euo pipefail

# Validate this script is running outside a container
if [ -e /.dockerenv ]; then
    echo "Error: This script must be run outside a Docker container"
    echo "This script builds dev images and should be run on the host system"
    exit 1
fi

# Validate docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH"
    echo "Please install Docker and ensure it's available in your PATH"
    exit 1
fi

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] <ci-image-name>

Convert a CI-built Docker image into a development-ready image.

OPTIONS:
    -h, --help, --h              Show this help message and exit
    --no-latest-tag              Skip tagging the dev image as 'dynamo:latest-vllm-local-dev'

ARGUMENTS:
    ci-image-name                The name of the CI image to convert
                                 (e.g. gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:66231cf0977716a60dc082c344f7e81a245929f3-32632154-vllm-amd64-dev)

EXAMPLES:
    $0 gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:66231cf0977716a60dc082c344f7e81a245929f3-32632154-vllm-amd64-dev
        Creates: dynamo:latest-vllm-dev

DESCRIPTION:
    This script takes an existing pre-built image from CI and converts it into a
    development-ready image by:
    - Installing development tools (git, clang, protobuf-compiler, etc.)
    - Setting up the ubuntu user with sudo access
    - Creating necessary directories for development
    - Configuring the environment for development work

    The resulting image will be tagged with '-dev' suffix and can be used
    in your devcontainer.json configuration.
EOF
}

# Parse command line arguments
LATEST_TAG=true
CI_BASE_IMAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help|--h)
            show_help
            exit 0
            ;;
        --no-latest-tag)
            LATEST_TAG=false
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
        *)
            if [ -z "$CI_BASE_IMAGE" ]; then
                CI_BASE_IMAGE="$1"
            else
                echo "Error: Multiple image names provided"
                echo "Use '$0 --help' for usage information"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$CI_BASE_IMAGE" ]; then
    echo "Error: No image name provided"
    echo "Use '$0 --help' for usage information"
    exit 1
fi

# Check if the image already has a -dev suffix
if [[ "$CI_BASE_IMAGE" == *"-dev"* ]]; then
    echo "Error: '$CI_BASE_IMAGE' already appears to be a dev image (contains '-dev')"
    echo "This script is for converting CI images to dev images, not dev images to dev images"
    exit 1
fi

set -x
# Strip host from URLs but keep the path for the dev image tag
if [[ "$CI_BASE_IMAGE" == *":"* && "$CI_BASE_IMAGE" == *"/"* ]]; then
    # Handle registry URLs (e.g., gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:latest)
    # Extract path and tag, remove host (including port)
    PATH_AND_TAG=$(echo "$CI_BASE_IMAGE" | sed 's/^[^/]*\///')
    DEV_IMAGE="${PATH_AND_TAG}-dev"
else
    DEV_IMAGE="${CI_BASE_IMAGE}-dev"
fi

# Extract RUST_VERSION from Dockerfile.vllm to keep in sync
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
DOCKERFILE_PATH="$SCRIPT_DIR/../container/Dockerfile.vllm"

if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Could not find $DOCKERFILE_PATH"
    exit 1
fi

RUST_VERSION=$(grep -E 'RUST_VERSION=' "$DOCKERFILE_PATH" | head -1 | sed 's/.*RUST_VERSION=//' | tr -d ' \\')
if [ -z "$RUST_VERSION" ]; then
    echo "Error: Could not extract RUST_VERSION from Dockerfile.vllm"
    echo "Make sure the Dockerfile contains: ENV RUST_VERSION=<version>"
    exit 1
fi

echo "Converting '$CI_BASE_IMAGE' to '$DEV_IMAGE' (Rust $RUST_VERSION)"

# Create a temporary Dockerfile
TEMP_DOCKERFILE=$(mktemp)
trap "rm -f $TEMP_DOCKERFILE" EXIT

cat > "$TEMP_DOCKERFILE" << EOF
FROM $CI_BASE_IMAGE

# Install development tools and packages
RUN apt-get update -y && \\
DEBIAN_FRONTEND=noninteractive \\
    apt-get install -y --no-install-recommends \\
    # Build tools \\
    git \\
    wget \\
    curl \\
    jq \\
    sudo \\
    file \\
    # Rust/C++ development \\
    clang \\
    libclang-dev \\
    protobuf-compiler && \\
rm -rf /var/lib/apt/lists/*

# Install uv from the official Docker image
# uv: Fast Python package installer and resolver
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create ubuntu user if it doesn't exist
RUN if ! id ubuntu > /dev/null 2>&1; then \\
        useradd -m -s /bin/bash ubuntu; \\
    fi

# Configure sudo for ubuntu user
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \\
    chmod 440 /etc/sudoers.d/ubuntu

# Copy and chmod. Run as the normal uid/gid of the user (not ubuntu) to make sure
# the permissions are the same between the host and the container.
COPY $SCRIPT_DIR/_build-dev-image-helper.sh /tmp/install-dev-tools.sh
RUN chmod +x /tmp/install-dev-tools.sh && RUST_VERSION=$RUST_VERSION UID=$(id -u) GID=$(id -g) /tmp/install-dev-tools.sh

# Set up development environment
WORKDIR /home/ubuntu
USER ubuntu

# Add cargo paths and maturin paths to PATH
ENV PATH="/usr/local/cargo/bin:/opt/dynamo/venv/bin:\$PATH"

# Create necessary directories
RUN mkdir -p /home/ubuntu/.cache/pre-commit

# Set default shell
CMD ["/bin/bash"]
EOF

set +x > /dev/null

docker build -f "$TEMP_DOCKERFILE" -t "$DEV_IMAGE" .

# Clean up dangling images
docker image prune -f

# Tag as latest local dev image (unless --no-latest-tag is specified)
if [ "$LATEST_TAG" = true ]; then
    LATEST_TAG_NAME="dynamo:latest-vllm-local-dev"
    docker tag "$DEV_IMAGE" "$LATEST_TAG_NAME"
    echo "Created: $DEV_IMAGE and $LATEST_TAG_NAME"
else
    echo "Created: $DEV_IMAGE"
fi

echo "Ready for use in devcontainer.json"
