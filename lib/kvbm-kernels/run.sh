#!/bin/bash

set -e

IMAGE_NAME="kvbm-kernel"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo ""
echo "Running container with GPU support..."
docker run --rm \
    --gpus all \
    "$IMAGE_NAME"
    "$@"
