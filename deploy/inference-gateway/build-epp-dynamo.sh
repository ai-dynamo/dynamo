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

# Configuration - Set these environment variables before running
if [[ -z "${DYNAMO_DIR}" ]]; then
    echo "DYNAMO_DIR environment variable must be set"
    echo "   Example: export DYNAMO_DIR=/path/to/dynamo"
    exit 1
fi

if [[ -z "${GAIE_DIR}" ]]; then
    echo "GAIE_DIR environment variable must be set"
    echo "   Example: export GAIE_DIR=/path/to/gateway-api-inference-extension"
    exit 1
fi
DYNAMO_LIB_DIR="${GAIE_DIR}/pkg/epp/scheduling/plugins/dynamo_kv_scorer/lib"
DYNAMO_INCLUDE_DIR="${GAIE_DIR}/pkg/epp/scheduling/plugins/dynamo_kv_scorer/include"

echo "Building Dynamo KV Router C Library..."

# Step 1: Build the static library
echo "Building static library..."
cd "${DYNAMO_DIR}"
cargo build --release -p libdynamo_llm

# Step 2: Generate header file (with fallback)
echo "Generating C header..."
HEADER_OUTPUT="${DYNAMO_DIR}/lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h"

if ! cbindgen --config lib/bindings/c/cbindgen.toml --crate libdynamo_llm --output "${HEADER_OUTPUT}"; then
    echo "cbindgen failed, using fallback header..."
    cp "${DYNAMO_DIR}/lib/bindings/c/src/fallback_header.h" "${HEADER_OUTPUT}"
fi

# Step 3: Ensure directories exist
echo "Preparing directories..."
mkdir -p "${DYNAMO_LIB_DIR}"
mkdir -p "${DYNAMO_INCLUDE_DIR}"

# Step 4: Copy files to GAIE project
echo "Copying files to the GAIE project..."
cp "${HEADER_OUTPUT}" "${DYNAMO_INCLUDE_DIR}/"
cp "${DYNAMO_DIR}/target/release/libdynamo_llm_capi.a" "${DYNAMO_LIB_DIR}/"
cp "${DYNAMO_DIR}/container/Dockerfile.epp" "${GAIE_DIR}/Dockerfile.dynamo"

# Verify files were copied
if [[ ! -f "${DYNAMO_INCLUDE_DIR}/llm_engine.h" ]]; then
    echo "Header file copy failed!"
    exit 1
fi

if [[ ! -f "${DYNAMO_LIB_DIR}/libdynamo_llm_capi.a" ]]; then
    echo "Library file copy failed!"
    exit 1
fi

if [[ ! -f "${GAIE_DIR}/Dockerfile.dynamo" ]]; then
    echo "Docker.dynamo file copy failed!"
    exit 1
fi

echo "Files copied successfully:"
echo "   Header: ${DYNAMO_INCLUDE_DIR}/llm_engine.h"
echo "   Library: ${DYNAMO_LIB_DIR}/libdynamo_llm_capi.a"
echo "   Docker: ${GAIE_DIR}/Dockerfile.epp"

# Step 5: Apply Dynamo patch (if it exists)
echo "Applying Dynamo patch..."
cd "${GAIE_DIR}"

PATCH_FILE="${DYNAMO_DIR}/deploy/inference-gateway/epp-patches/v0.5.1-2/epp-v0.5.1-dyn2.patch"
if [[ -f "${PATCH_FILE}" ]]; then
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        echo "Patch applied successfully"
    else
        echo "Patch doesn't apply cleanly - may already be applied or need manual resolution"
    fi
else
    echo "No patch file found at ${PATCH_FILE}"
fi

# Step 6: Build the EPP image
echo "Building the custom EPP image for GAIE..."
make dynamo-image-local-load

echo "EPP image with Dynamo KV routing built"
