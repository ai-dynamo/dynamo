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
# REMOVE nvshmem cherry-pick when moving to next version of vllm
VLLM_REF="1da94e673c257373280026f75ceb4effac80e892"  # from v0.10.1.1
# When updating above VLLM_REF make sure precompiled wheel file URL is correct. Run this command:
# aws s3 ls s3://vllm-wheels/${VLLM_REF}/ --region us-west-2 --no-sign-request
VLLM_PRECOMPILED_WHEEL_LOCATION="https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_REF}/vllm-0.10.1.1-cp38-abi3-manylinux1_x86_64.whl"
VLLM_GIT_URL="https://github.com/vllm-project/vllm.git"
MAX_JOBS=16
INSTALLATION_DIR=/tmp
ARCH=$(uname -m)
DEEPGEMM_REF="f85ec64"
FLASHINF_REF="v0.2.11"
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
        --vllm-git-url)
            VLLM_GIT_URL="$2"
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
            echo "  --vllm-ref REF    Git reference to checkout (default: ${VLLM_REF})"
            echo "  --max-jobs NUM    Maximum number of parallel jobs (default: ${MAX_JOBS})"
            echo "  --arch ARCH       Architecture (amd64|arm64, default: auto-detect)"
            echo "  --installation-dir DIR  Directory to install vllm (default: ${INSTALLATION_DIR})"
            echo "  --deepgemm-ref REF  Git reference for DeepGEMM (default: ${DEEPGEMM_REF})"
            echo "  --flashinf-ref REF  Git reference for Flash Infer (default: ${FLASHINF_REF})"
            echo "  --torch-backend BACKEND  Torch backend to use (default: ${TORCH_BACKEND})"
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

if [ "$ARCH" = "amd64" ]; then
    # LMCache installation currently fails on arm64 due to CUDA dependency issues:
    # OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
    # TODO: Re-enable for arm64 after verifying lmcache compatibility and resolving the build issue.
    uv pip install lmcache==0.3.3
fi

# Create vllm directory and clone
mkdir -p $INSTALLATION_DIR
cd $INSTALLATION_DIR
git clone $VLLM_GIT_URL vllm
cd vllm
git checkout $VLLM_REF
# nvshmem fix - cherry-pick commit pinning pplx version
# https://github.com/ai-dynamo/dynamo/actions/runs/17907241473/job/50910654042?pr=2969#step:8:280
# remove when moving to next version of vllm
# Configure git user for cherry-pick operation
GIT_COMMITTER_NAME="Container Build" GIT_COMMITTER_EMAIL="container@buildkitsandbox.local" git cherry-pick 906e461ed6ddccd3cc7b68fa72048d2d3fcbd72c

if [ "$ARCH" = "arm64" ]; then
    echo "Installing vllm for ARM64 architecture"

    # Try to install specific PyTorch version first, fallback to latest nightly
    echo "Attempting to install pinned PyTorch nightly versions..."
    if ! uv pip install torch==2.7.1+cu128 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl; then
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

    echo "Attempting to install pinned OpenAI version..."
    if ! uv pip install  openai==1.99.9; then
        echo "Pinned versions failed"
        exit 1
    fi

    export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION}"

    if [ "$EDITABLE" = "true" ]; then
	uv pip install -e . --torch-backend=$TORCH_BACKEND
    else
        uv pip install . --torch-backend=$TORCH_BACKEND
    fi
fi

# Install ep_kernels and DeepGEMM
echo "Installing ep_kernels and DeepGEMM"
cd tools/ep_kernels
TORCH_CUDA_ARCH_LIST="9.0;10.0" bash install_python_libraries.sh # These libraries aren't pinned.
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
