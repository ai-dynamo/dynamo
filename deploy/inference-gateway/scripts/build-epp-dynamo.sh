#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration - DYNAMO_DIR defaults to the parent dynamo repo
if [[ -z "${DYNAMO_DIR}" ]]; then
    # Default: assume we're in dynamo/deploy/inference-gateway
    DYNAMO_DIR="$(cd "${PROJECT_DIR}/../.." && pwd)"
    echo "DYNAMO_DIR not set, using default: ${DYNAMO_DIR}"
fi

# Verify we're in the right place
if [[ ! -f "${PROJECT_DIR}/Makefile" ]]; then
    echo "ERROR: Makefile not found in ${PROJECT_DIR}"
    exit 1
fi

if [[ ! -d "${DYNAMO_DIR}/lib/bindings/c" ]]; then
    echo "ERROR: Dynamo source not found at ${DYNAMO_DIR}"
    echo "Set DYNAMO_DIR to the root of the dynamo repository"
    exit 1
fi

echo "============================================="
echo "Dynamo EPP Build Script"
echo "============================================="
echo "Project Dir: ${PROJECT_DIR}"
echo "Dynamo Dir:  ${DYNAMO_DIR}"
echo "============================================="

cd "${PROJECT_DIR}"

# Build mode: local (default), push, or kind
BUILD_MODE="${1:-local}"

case "${BUILD_MODE}" in
    local)
        echo "Building Dynamo library and Docker image (load locally)..."
        make all DYNAMO_DIR="${DYNAMO_DIR}"
        ;;
    push)
        echo "Building Dynamo library and Docker image (push to registry)..."
        make all-push DYNAMO_DIR="${DYNAMO_DIR}"
        ;;
    kind)
        echo "Building Dynamo library and Docker image (load to kind)..."
        make all-kind DYNAMO_DIR="${DYNAMO_DIR}"
        ;;
    lib-only)
        echo "Building Dynamo library only..."
        make dynamo-lib DYNAMO_DIR="${DYNAMO_DIR}"
        ;;
    image-only)
        echo "Building Docker image only (library must exist)..."
        make image-local-load DYNAMO_DIR="${DYNAMO_DIR}"
        ;;
    *)
        echo "Usage: $0 [local|push|kind|lib-only|image-only]"
        echo ""
        echo "Modes:"
        echo "  local      - Build library and image, load locally (default)"
        echo "  push       - Build library and image, push to registry"
        echo "  kind       - Build library and image, load to kind cluster"
        echo "  lib-only   - Build Dynamo library only"
        echo "  image-only - Build Docker image only (library must exist)"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "Build complete!"
echo "============================================="
make info DYNAMO_DIR="${DYNAMO_DIR}"
