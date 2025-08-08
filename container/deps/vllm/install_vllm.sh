#!/usr/bin/env bash
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

set -euo pipefail

# Parse arguments
EDITABLE=true
VLLM_REF="f4135232b9a8c4845f8961fb1cd17581c56ae2ce"
MAX_JOBS=16
INSTALLATION_DIR=/tmp
ARCH=$(uname -m)
DEEPGEMM_REF="03d0be3"
FLASHINF_REF="v0.2.8rc1"
TORCH_BACKEND="cu128"

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
        --installation-dir)
            INSTALLATION_DIR="$2"
            shift 2
            ;;
        --deepgemm-ref)
            DEEPGEMM_REF="$2"
            shift 2
            ;;
        --flashinf-ref)
            FLASHINF_REF="$2"
            shift 2
            ;;
        --torch-backend)
            TORCH_BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--editable|--no-editable] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--deepgemm-ref REF] [--flashinf-ref REF] [--torch-backend BACKEND]"
            echo "Options:"
            echo "  --editable        Install vllm in editable mode (default)"
            echo "  --no-editable     Install vllm in non-editable mode"
            echo "  --vllm-ref REF    Git reference to checkout (default: f4135232b9a8c4845f8961fb1cd17581c56ae2ce)"
            echo "  --max-jobs NUM    Maximum number of parallel jobs (default: 16)"
            echo "  --arch ARCH       Architecture (amd64|arm64, default: auto-detect)"
            echo "  --installation-dir DIR  Directory to install vllm (default: /tmp/vllm)"
            echo "  --deepgemm-ref REF  Git reference for DeepGEMM (default: 1876566)"
            echo "  --flashinf-ref REF  Git reference for Flash Infer (default: v0.2.8rc1)"
            echo "  --torch-backend BACKEND  Torch backend to use (default: cu128)"
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
echo "  TORCH_BACKEND: $TORCH_BACKEND"

# Install common dependencies
uv pip install pip cuda-python

# Install LMCache
uv pip install lmcache

# Create vllm directory and clone
mkdir -p $INSTALLATION_DIR
cd $INSTALLATION_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout $VLLM_REF

if [ "$ARCH" = "arm64" ]; then
    echo "Installing vllm for ARM64 architecture"

    # Try to install specific PyTorch version first, fallback to latest nightly
    echo "Attempting to install pinned PyTorch nightly versions..."
    if ! uv pip install torch==2.8.0.dev20250613+cu128 torchaudio==2.8.0.dev20250616 torchvision==0.23.0.dev20250616 --index-url https://download.pytorch.org/whl/nightly/cu128; then
        echo "Pinned versions failed"
        exit 1
        # uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    fi

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
        VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=$TORCH_BACKEND
    else
        VLLM_USE_PRECOMPILED=1 uv pip install . --torch-backend=$TORCH_BACKEND
    fi
fi

# Install ep_kernels and DeepGEMM
echo "Installing ep_kernels and DeepGEMM"
cd tools/ep_kernels
bash install_python_libraries.sh # These libraries aren't pinned.
cd ep_kernels_workspace
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
git checkout $DEEPGEMM_REF # Pin Version

sed -i 's|git@github.com:|https://github.com/|g' .gitmodules
git submodule sync --recursive
git submodule update --init --recursive

# command for 03d0be3
python setup.py install

# new install command for post 03d0be3
# cat install.sh
# ./install.sh


# Install Flash Infer
if [ "$ARCH" = "arm64" ]; then
    uv pip install flashinfer-python
else
    cd $INSTALLATION_DIR
    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer
    git checkout $FLASHINF_REF
    uv pip install -v --no-build-isolation .
fi

echo "vllm installation completed successfully"
