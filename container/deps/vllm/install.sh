#!/bin/bash
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

set -e

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --patch PATH        Apply a patch file during installation"
    echo "  --ref REF          Specify the vLLM git reference (branch/tag/commit) to install"
    echo "  --install-cmd CMD  Specify the installation command (default: 'pip install')"
    echo "  --use-precompiled  Use precompiled kernels during installation"
    echo "  --installation-dir DIR  Specify the installation directory (default: 'vllm')"
    echo "  --help             Show this help message"
}

# Default values
INSTALL_CMD="pip install"
VLLM_REF="main"
PATCH_PATH=""
USE_PRECOMPILED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch)
            PATCH_PATH="$2"
            shift 2
            ;;
        --ref)
            VLLM_REF="$2"
            shift 2
            ;;
        --install-cmd)
            INSTALL_CMD="$2"
            shift 2
            ;;
        --use-precompiled)
            USE_PRECOMPILED=true
            shift
            ;;
        --installation-dir)
            INSTALLATION_DIR="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create temp directory and clean it up on exit

# Convert patch path to absolute path if it's relative
if [[ ! "$PATCH_PATH" = /* ]]; then
    PATCH_PATH="$(pwd)/${PATCH_PATH}"
fi

# Clone vLLM repository
echo "Cloning vLLM repository at ref: $VLLM_REF"
git clone https://github.com/vllm-project/vllm.git "$INSTALLATION_DIR"
cd "$INSTALLATION_DIR"
git checkout "$VLLM_REF"

# Apply patch if provided
if [ -n "$PATCH_PATH" ]; then
    echo "Applying patch from: $PATCH_PATH"
    git apply "$PATCH_PATH"
fi

# Install using specified command
echo "Installing using: $INSTALL_CMD"
if [ "$USE_PRECOMPILED" = true ]; then
    echo "Using precompiled kernels"
    export VLLM_PRECOMPILED_WHEEL_LOCATION="https://files.pythonhosted.org/packages/e7/c0/5b7f019aa798dedfb44c30971e9becf3c6a2db7dde311570178fa66c49c8/vllm-0.7.2-cp38-abi3-manylinux1_x86_64.whl"
fi
$INSTALL_CMD .

echo "Installation complete!"
