#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script installs vLLM and its dependencies from PyPI (release versions only).
# Installation order:
# 1. LMCache (installed first so vLLM's dependencies take precedence)
# 2. vLLM
# 3. DeepGEMM
# 4. EP kernels

set -euo pipefail

VLLM_REF="v0.11.2"

# Basic Configurations
ARCH=$(uname -m)
MAX_JOBS=16
INSTALLATION_DIR=/tmp

# VLLM and Dependency Configurations
TORCH_CUDA_ARCH_LIST="9.0;10.0" # For EP Kernels
DEEPGEMM_REF=""
CUDA_VERSION="12.9"
FLASHINF_REF="v0.5.2"
# LMCache version - 0.3.9+ required for vLLM 0.11.2 compatibility
LMCACHE_REF="0.3.9.post2"

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --lmcache-ref)
            LMCACHE_REF="$2"
            shift 2
            ;;
        --torch-cuda-arch-list)
            TORCH_CUDA_ARCH_LIST="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--deepgemm-ref REF] [--flashinf-ref REF] [--lmcache-ref REF] [--torch-cuda-arch-list LIST] [--cuda-version VERSION]"
            echo "Options:"
            echo "  --vllm-ref REF      vLLM release version (default: ${VLLM_REF})"
            echo "  --max-jobs NUM      Maximum parallel jobs (default: ${MAX_JOBS})"
            echo "  --arch ARCH         Architecture amd64|arm64 (default: auto-detect)"
            echo "  --installation-dir DIR  Install directory (default: ${INSTALLATION_DIR})"
            echo "  --deepgemm-ref REF  DeepGEMM git ref (default: ${DEEPGEMM_REF})"
            echo "  --flashinf-ref REF  FlashInfer version (default: ${FLASHINF_REF})"
            echo "  --lmcache-ref REF   LMCache version (default: ${LMCACHE_REF})"
            echo "  --torch-cuda-arch-list LIST  CUDA architectures (default: ${TORCH_CUDA_ARCH_LIST})"
            echo "  --cuda-version VERSION  CUDA version (default: ${CUDA_VERSION})"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert x86_64 to amd64 for consistency with Docker ARG
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

export MAX_JOBS=$MAX_JOBS
export CUDA_HOME=/usr/local/cuda

# Derive torch backend from CUDA version (e.g., "12.9" -> "cu129")
TORCH_BACKEND="cu$(echo $CUDA_VERSION | tr -d '.')"

echo "=== Installing prerequisites ==="
uv pip install pip cuda-python

echo "\n=== Configuration Summary ==="
echo "  VLLM_REF=$VLLM_REF | ARCH=$ARCH | CUDA_VERSION=$CUDA_VERSION | TORCH_BACKEND=$TORCH_BACKEND"
echo "  FLASHINF_REF=$FLASHINF_REF | LMCACHE_REF=$LMCACHE_REF | DEEPGEMM_REF=$DEEPGEMM_REF"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST | INSTALLATION_DIR=$INSTALLATION_DIR"

echo "\n=== Installing LMCache ==="
if [ "$ARCH" = "amd64" ]; then
    # LMCache installation currently fails on arm64 due to CUDA dependency issues
    # Install LMCache BEFORE vLLM so vLLM's dependencies take precedence
    uv pip install lmcache==${LMCACHE_REF} --torch-backend=${TORCH_BACKEND}
    echo "✓ LMCache ${LMCACHE_REF} installed"
else
    echo "⚠ Skipping LMCache on ARM64 (compatibility issues)"
fi

echo "\n=== Cloning vLLM repository ==="
# Clone needed for DeepGEMM and EP kernels install scripts
cd $INSTALLATION_DIR
git clone https://github.com/vllm-project/vllm.git vllm
cd vllm
git checkout $VLLM_REF

echo "\n=== Installing vLLM & FlashInfer ==="
echo "Installing vLLM $VLLM_REF from PyPI..."

uv pip install vllm[flashinfer]==$VLLM_REF --torch-backend=${TORCH_BACKEND}
uv pip install flashinfer-cubin==$FLASHINF_REF
uv pip install flashinfer-jit-cache==$FLASHINF_REF --extra-index-url https://flashinfer.ai/whl/${TORCH_BACKEND}

echo "✓ vLLM installation completed"

echo "\n=== Installing DeepGEMM ==="
cd $INSTALLATION_DIR/vllm/tools

if [ -n "$DEEPGEMM_REF" ]; then
    bash install_deepgemm.sh --cuda-version "${CUDA_VERSION}" --ref "$DEEPGEMM_REF"
else
    bash install_deepgemm.sh --cuda-version "${CUDA_VERSION}"
fi
echo "✓ DeepGEMM installation completed"

echo "\n=== Installing EP Kernels (PPLX and DeepEP) ==="
cd ep_kernels/
# TODO we will be able to specify which pplx and deepep commit we want in future
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" bash install_python_libraries.sh

echo "\n✅ All installations completed successfully!"
