#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script installs vLLM and its dependencies from PyPI (release versions only).
# Installation order:
# 1. vLLM
# 2. LMCache (built from source AFTER vLLM so c_ops.so is compiled against installed PyTorch)
# 3. vLLM-Omni
# 4. DeepGEMM
# 5. EP kernels

set -euo pipefail

VLLM_VER="0.19.0"
VLLM_REF="v${VLLM_VER}"
DEVICE="cuda"

# Basic Configurations
ARCH=$(uname -m)
MAX_JOBS=16
INSTALLATION_DIR=/tmp

# VLLM and Dependency Configurations
TORCH_CUDA_ARCH_LIST="9.0;10.0" # For EP Kernels -- TODO: check if we need to add 12.0+PTX
DEEPGEMM_REF=""
CUDA_VERSION="12.9"
FLASHINF_REF="v0.6.6"
LMCACHE_REF="0.4.2"
VLLM_OMNI_REF="v0.16.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
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
        --lmcache-ref)
            LMCACHE_REF="$2"
            shift 2
            ;;
        --vllm-omni-ref)
            VLLM_OMNI_REF="$2"
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
            echo "Usage: $0 [--device DEVICE] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--deepgemm-ref REF] [--flashinf-ref REF] [--lmcache-ref REF] [--vllm-omni-ref REF] [--torch-cuda-arch-list LIST] [--cuda-version VERSION]"
            echo "Options:"
            echo "  --device DEVICE     Device Selection (default: cuda)"
            echo "  --vllm-ref REF      vLLM release version (default: ${VLLM_REF})"
            echo "  --max-jobs NUM      Maximum parallel jobs (default: ${MAX_JOBS})"
            echo "  --arch ARCH         Architecture amd64|arm64 (default: auto-detect)"
            echo "  --installation-dir DIR  Install directory (default: ${INSTALLATION_DIR})"
            echo "  --deepgemm-ref REF  DeepGEMM git ref (default: ${DEEPGEMM_REF})"
            echo "  --flashinf-ref REF  FlashInfer version (default: ${FLASHINF_REF})"
            echo "  --lmcache-ref REF   LMCache version (default: ${LMCACHE_REF})"
            echo "  --vllm-omni-ref REF vLLM-Omni version (default: ${VLLM_OMNI_REF})"
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

# Set alternative CPU architecture naming
if [ "$ARCH" = "amd64" ]; then
    ALT_ARCH="x86_64"
elif [ "$ARCH" = "arm64" ]; then
    ALT_ARCH="aarch64"
fi

export MAX_JOBS=$MAX_JOBS
if [ "$DEVICE" = "cuda" ]; then
    export CUDA_HOME=/usr/local/cuda

    # Derive torch backend from CUDA version (e.g., "12.9" -> "cu129")
    TORCH_BACKEND="cu$(echo $CUDA_VERSION | tr -d '.')"
    CUDA_VERSION_MAJOR=${CUDA_VERSION%%.*}

    echo "=== Installing prerequisites ==="
    uv pip install pip cuda-python
fi

if [ "$DEVICE" = "cuda" ]; then
    echo "\n=== Configuration Summary ==="
    echo "  VLLM_REF=$VLLM_REF | ARCH=$ARCH | CUDA_VERSION=$CUDA_VERSION | TORCH_BACKEND=$TORCH_BACKEND"
    echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST | INSTALLATION_DIR=$INSTALLATION_DIR"
elif [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then
    echo "\n=== Configuration Summary ==="
    echo "  VLLM_REF=$VLLM_REF | ARCH=$ARCH | INSTALLATION_DIR=$INSTALLATION_DIR"
fi

echo "\n=== Cloning vLLM repository ==="
# Clone needed for DeepGEMM and EP kernels install scripts
cd $INSTALLATION_DIR
git clone https://github.com/vllm-project/vllm.git vllm
cd vllm
git checkout $VLLM_REF
echo "✓ vLLM repository cloned"

echo "\n=== Installing vLLM-Omni ==="
# Install omni BEFORE vLLM. Its transitive dependencies can otherwise upgrade the
# torch/transformers stack after vLLM is installed, which can leave vllm._C ABI-mismatched.
# vLLM should remain the final owner of the runtime stack in this environment.
if [ -n "$VLLM_OMNI_REF" ] && [ "$ARCH" = "amd64" ]; then
    # Try PyPI first, fall back to building from source
    if uv pip install vllm-omni==${VLLM_OMNI_REF#v} 2>&1; then
        echo "✓ vLLM-Omni ${VLLM_OMNI_REF} installed from PyPI"
    else
        echo "⚠ PyPI install failed, building from source..."
        git clone --depth 1 --branch ${VLLM_OMNI_REF} https://github.com/vllm-project/vllm-omni.git $INSTALLATION_DIR/vllm-omni
        uv pip install $INSTALLATION_DIR/vllm-omni
        rm -rf $INSTALLATION_DIR/vllm-omni
        echo "✓ vLLM-Omni ${VLLM_OMNI_REF} installed from source"
    fi
else
    echo "⚠ Skipping vLLM-Omni (no ref provided or ARM64 not supported)"
fi

if [ "$DEVICE" = "xpu" ]; then
    echo "\n=== Installing vLLM ==="
    uv pip install -r requirements/xpu.txt --index-strategy unsafe-best-match
    uv pip install --verbose --no-build-isolation .
fi

if [ "$DEVICE" = "cuda" ]; then
    echo "\n=== Installing vLLM & FlashInfer ==="

    # Build GitHub release wheel URL per CUDA version
    # CUDA 12 wheels have no +cu suffix and use manylinux_2_31
    # CUDA 13 wheels have +cu130 suffix and use manylinux_2_35
    if [[ "$CUDA_VERSION_MAJOR" == "12" ]]; then
        VLLM_GITHUB_WHEEL="vllm-${VLLM_VER}-cp38-abi3-manylinux_2_31_${ALT_ARCH}.whl"
        EXTRA_PIP_ARGS=""
    elif [[ "$CUDA_VERSION_MAJOR" == "13" ]]; then
        VLLM_GITHUB_WHEEL="vllm-${VLLM_VER}+${TORCH_BACKEND}-cp38-abi3-manylinux_2_35_${ALT_ARCH}.whl"
        EXTRA_PIP_ARGS="--index-strategy=unsafe-best-match --extra-index-url https://download.pytorch.org/whl/${TORCH_BACKEND}"
    else
        echo "❌ Unsupported CUDA version for vLLM installation: ${CUDA_VERSION}"
        exit 1
    fi
    VLLM_GITHUB_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/${VLLM_GITHUB_WHEEL}"

    # Install vLLM wheel
    # CUDA 12: Try PyPI first, fall back to GitHub release
    # CUDA 13: Always use GitHub release (PyPI only has cu12 wheels, --torch-backend
    #           does not prevent uv from resolving the cu12 variant)
    echo "Installing vLLM $VLLM_VER (torch backend: $TORCH_BACKEND)..."
    if [[ "$CUDA_VERSION_MAJOR" == "12" ]]; then
        if uv pip install "vllm[flashinfer,runai,otel]==${VLLM_VER}" ${EXTRA_PIP_ARGS} --torch-backend=${TORCH_BACKEND} 2>&1; then
            echo "✓ vLLM ${VLLM_VER} installed from PyPI"
        else
            echo "⚠ PyPI install failed, installing from GitHub release..."
            uv pip install ${EXTRA_PIP_ARGS} \
                "${VLLM_GITHUB_URL}[flashinfer,runai,otel]" \
                --torch-backend=${TORCH_BACKEND}
            echo "✓ vLLM ${VLLM_VER} installed from GitHub"
        fi
    else
        echo "Installing vLLM from GitHub release (cu130 wheel not available on PyPI)..."
        uv pip install ${EXTRA_PIP_ARGS} \
            "${VLLM_GITHUB_URL}[flashinfer,runai,otel]" \
            --torch-backend=${TORCH_BACKEND}
        echo "✓ vLLM ${VLLM_VER} installed from GitHub"
    fi
    uv pip install flashinfer-cubin==$FLASHINF_REF
    uv pip install flashinfer-jit-cache==$FLASHINF_REF --extra-index-url https://flashinfer.ai/whl/${TORCH_BACKEND}
fi

if [ "$DEVICE" = "cpu" ]; then
    echo "\n=== Installing vLLM for cpu ==="
    if [ -n "${CACHE_BUSTER:-}" ]; then
        echo "$CACHE_BUSTER" > /tmp/builder-buster
    fi
    # vLLM CPU requirements pin torch with a +cpu local version (e.g. 2.10.0+cpu),
    # which is published on the PyTorch CPU wheel index instead of PyPI.
    # Install torchvision, torchaudio from the same index to get the correct versions with +cpu suffix.
    uv pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    uv pip install torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    VLLM_TARGET_DEVICE=cpu \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
    uv pip install dist/*.whl
fi
echo "✓ vLLM installation completed"

echo "\n=== Installing LMCache from source ==="
# LMCache prebuilt wheels are built against PyTorch <=2.8.0 and fail with PyTorch 2.10+
# (undefined symbol: c10::cuda::c10_cuda_check_implementation).
# Build from source AFTER vLLM so c_ops.so compiles against the installed PyTorch.
# Ref: https://docs.lmcache.ai/getting_started/installation.html#install-latest-lmcache-from-source
if [ "$DEVICE" = "cuda" ]; then
    git clone --depth 1 --branch v${LMCACHE_REF} https://github.com/LMCache/LMCache.git ${INSTALLATION_DIR}/lmcache
    cd ${INSTALLATION_DIR}/lmcache
    uv pip install -r requirements/build.txt
    # Get torch lib dir and embed it as RPATH so c_ops.so finds torch libs at runtime
    TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
    # Build from source with --no-build-isolation (uses installed torch) + RPATH for runtime linking
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0+PTX" LDFLAGS="-Wl,-rpath,${TORCH_LIB}" \
        uv pip install --no-build-isolation --no-cache .
    # Verify c_ops.so was compiled (cannot import at build time without GPU/CUDA driver)
    # cd to neutral dir so Python finds installed lmcache, not the source checkout
    cd /tmp
    LMCACHE_DIR=$(python3 -c "import lmcache, os; print(os.path.dirname(lmcache.__file__))")
    if ls "${LMCACHE_DIR}"/c_ops*.so > /dev/null 2>&1; then
        echo "✓ lmcache c_ops.so verified: $(ls ${LMCACHE_DIR}/c_ops*.so | head -1 | xargs basename)"
    else
        echo "ERROR: c_ops.so not found in ${LMCACHE_DIR} - CUDA extension was not compiled"
        exit 1
    fi
    rm -rf ${INSTALLATION_DIR}/lmcache
    echo "✓ LMCache ${LMCACHE_REF} installed from source"
elif [ "$DEVICE" = "xpu" ] && [ "$ARCH" = "amd64" ]; then
    uv pip install lmcache==${LMCACHE_REF}
    echo "✓ LMCache ${LMCACHE_REF} installed from PyPI (XPU)"
else
    echo "⚠ Skipping LMCache for DEVICE=${DEVICE} ARCH=${ARCH} (not supported)"
fi

if [ "$DEVICE" = "cuda" ]; then
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
fi
echo "\n✅ All installations completed successfully!"
