#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a Dynamo image with the RIVA NIM Python client layered on top, for the
# cascaded voice pipeline example.
#
#   BASE_IMAGE  Dynamo image to layer on (default: dynamo:latest-vllm-runtime)
#   TAG         Output image tag        (default: dynamo-riva:latest)

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-dynamo:latest-vllm-runtime}"
TAG="${TAG:-dynamo-riva:latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Building ${TAG} from base ${BASE_IMAGE}"
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${EXAMPLE_DIR}"

echo "Built ${TAG}"
