#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install vllm and wideEP kernels from a specific git reference

set -ex

# Parse arguments
EDITABLE=true
VLLM_REF="059d4cd"
MAX_JOBS=16
ARCH=$(uname -m)

# Convert x86_64 to amd64 for consistency with Docker ARG
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --editable)
            EDITABLE=true
            shift
            ;;
        --no-editable)
            EDITABLE=false
            shift
            ;;
        --vllm-ref)
            VLLM_REF="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--editable|--no-editable] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH]"
            echo "Options:"
            echo "  --editable        Install vllm in editable mode (default)"
            echo "  --no-editable     Install vllm in non-editable mode"
            echo "  --vllm-ref REF    Git reference to checkout (default: 059d4cd)"
            echo "  --max-jobs NUM    Maximum number of parallel jobs (default: 16)"
            echo "  --arch ARCH       Architecture (amd64|arm64, default: auto-detect)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export MAX_JOBS=$MAX_JOBS
export CUDA_HOME=/usr/local/cuda

echo "Installing vllm with the following configuration:"
echo "  EDITABLE: $EDITABLE"
echo "  VLLM_REF: $VLLM_REF"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  ARCH: $ARCH"

# Install common dependencies
uv pip install pip cuda-python

# Create vllm directory and clone
mkdir -p /opt/vllm
cd /opt/vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout $VLLM_REF

if [ "$ARCH" = "arm64" ]; then
    echo "Installing vllm for ARM64 architecture"
    uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    python use_existing_torch.py
    uv pip install -r requirements/build.txt

    if [ "$EDITABLE" = "true" ]; then
        MAX_JOBS=${MAX_JOBS} uv pip install --no-build-isolation -e . -v
    else
        MAX_JOBS=${MAX_JOBS} uv pip install --no-build-isolation . -v
    fi
else
    echo "Installing vllm for AMD64 architecture"
    if [ "$EDITABLE" = "true" ]; then
        VLLM_USE_PRECOMPILED=1 uv pip install -e .
    else
        VLLM_USE_PRECOMPILED=1 uv pip install .
    fi
fi

# Install ep_kernels and DeepGEMM
echo "Installing ep_kernels and DeepGEMM"
cd tools/ep_kernels
bash install_python_libraries.sh
cd ep_kernels_workspace
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
sed -i 's|git@github.com:|https://github.com/|g' .gitmodules
git submodule sync --recursive
git submodule update --init --recursive
cat install.sh
./install.sh

echo "vllm installation completed successfully"
