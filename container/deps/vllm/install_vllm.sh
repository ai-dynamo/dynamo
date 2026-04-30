#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script installs vLLM and its dependencies either from release wheels or from source.
# Installation order:
# 1. PyTorch for the requested CUDA backend
# 2. vLLM-Omni
# 3. vLLM
# 4. LMCache (built from source AFTER vLLM so c_ops.so is compiled against installed PyTorch)
# 5. DeepGEMM
# 6. EP kernels

set -euo pipefail

VLLM_INSTALL_MODE="wheel"
VLLM_VER="0.20.0"
VLLM_REF="v${VLLM_VER}"
VLLM_SOURCE_REF=""
DEVICE="cuda"

# Basic Configurations
ARCH=$(uname -m)
MAX_JOBS=16
INSTALLATION_DIR=/tmp

# VLLM and Dependency Configurations
TORCH_CUDA_ARCH_LIST="9.0;10.0" # For EP Kernels -- TODO: check if we need to add 12.0+PTX
DEEPGEMM_REF=""
CUDA_VERSION="12.9"
FLASHINF_REF="v0.6.8.post1"
LMCACHE_REF="0.4.4"
VLLM_OMNI_REF="release/v0.19.0rc1"
TORCH_REF="2.11.0"
TORCHVISION_REF="0.26.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --vllm-install-mode)
            VLLM_INSTALL_MODE="$2"
            shift 2
            ;;
        --vllm-version)
            # Backward compatibility for Shenron-generated stable Dockerfiles.
            VLLM_VER="${2#v}"
            VLLM_REF="v${VLLM_VER}"
            shift 2
            ;;
        --vllm-source-ref)
            VLLM_SOURCE_REF="$2"
            shift 2
            ;;
        --vllm-ref)
            VLLM_REF="$2"
            VLLM_VER="${VLLM_REF#v}"
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
            echo "Usage: $0 [--device DEVICE] [--vllm-install-mode MODE] [--vllm-version VERSION] [--vllm-source-ref REF] [--vllm-ref REF] [--max-jobs NUM] [--arch ARCH] [--deepgemm-ref REF] [--flashinf-ref REF] [--lmcache-ref REF] [--vllm-omni-ref REF] [--torch-cuda-arch-list LIST] [--cuda-version VERSION]"
            echo "Options:"
            echo "  --device DEVICE     Device Selection (default: cuda)"
            echo "  --vllm-install-mode MODE  Install mode: wheel|source (default: ${VLLM_INSTALL_MODE})"
            echo "  --vllm-version VERSION    Compatibility alias for stable Shenron builds (default: ${VLLM_VER})"
            echo "  --vllm-source-ref REF     vLLM git ref for source installs (default: unset)"
            echo "  --vllm-ref REF      vLLM release ref for wheel installs (default: ${VLLM_REF})"
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

if [ "$VLLM_INSTALL_MODE" != "wheel" ] && [ "$VLLM_INSTALL_MODE" != "source" ]; then
    echo "Unknown vLLM install mode: $VLLM_INSTALL_MODE"
    exit 1
fi

if [ "$VLLM_INSTALL_MODE" = "source" ]; then
    if [ -z "$VLLM_SOURCE_REF" ]; then
        echo "Source install mode requires --vllm-source-ref"
        exit 1
    fi
    VLLM_CHECKOUT_REF="$VLLM_SOURCE_REF"
else
    VLLM_CHECKOUT_REF="$VLLM_REF"
fi

# Convert x86_64 to amd64 for consistency with Docker ARG
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi

export MAX_JOBS=$MAX_JOBS
TORCH_UV_ARGS=""
VLLM_UV_ARGS=""
if [ "$DEVICE" = "cuda" ]; then
    export CUDA_HOME=/usr/local/cuda

    # Derive torch backend from CUDA version (e.g., "12.9" -> "cu129")
    TORCH_BACKEND="cu$(echo $CUDA_VERSION | tr -d '.')"
    CUDA_VERSION_MAJOR=${CUDA_VERSION%%.*}
    CUDA_VERSION_MINOR=$(echo "${CUDA_VERSION#*.}" | cut -d. -f1)
    if [[ "$TORCH_BACKEND" == "cu129" || "$TORCH_BACKEND" == "cu130" ]]; then
        TORCH_UV_ARGS="--index-url https://pypi.org/simple --index-strategy=unsafe-best-match --torch-backend=${TORCH_BACKEND}"
        VLLM_UV_ARGS="${TORCH_UV_ARGS} --extra-index-url https://wheels.vllm.ai/${VLLM_VER}/${TORCH_BACKEND}"
    else
        echo "❌ Unsupported CUDA version for vLLM installation: ${CUDA_VERSION}"
        exit 1
    fi

    echo "=== Installing prerequisites ==="
    uv pip install ${TORCH_UV_ARGS} pip cuda-python
    echo "Installing PyTorch ${TORCH_REF} for ${TORCH_BACKEND}..."
    uv pip install ${TORCH_UV_ARGS} \
        "torch==${TORCH_REF}+${TORCH_BACKEND}" \
        "torchaudio==${TORCH_REF}+${TORCH_BACKEND}" \
        "torchvision==${TORCHVISION_REF}+${TORCH_BACKEND}"
fi

if [ "$DEVICE" = "cuda" ]; then
    echo "\n=== Configuration Summary ==="
    echo "  VLLM_INSTALL_MODE=$VLLM_INSTALL_MODE | VLLM_VER=$VLLM_VER | VLLM_REF=$VLLM_REF | VLLM_SOURCE_REF=${VLLM_SOURCE_REF:-unset}"
    echo "  VLLM_CHECKOUT_REF=$VLLM_CHECKOUT_REF | ARCH=$ARCH | CUDA_VERSION=$CUDA_VERSION | TORCH_BACKEND=$TORCH_BACKEND"
    echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST | INSTALLATION_DIR=$INSTALLATION_DIR"
elif [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then
    echo "\n=== Configuration Summary ==="
    echo "  VLLM_INSTALL_MODE=$VLLM_INSTALL_MODE | VLLM_VER=$VLLM_VER | VLLM_REF=$VLLM_REF | VLLM_SOURCE_REF=${VLLM_SOURCE_REF:-unset}"
    echo "  VLLM_CHECKOUT_REF=$VLLM_CHECKOUT_REF | ARCH=$ARCH | INSTALLATION_DIR=$INSTALLATION_DIR"
fi

echo "\n=== Cloning vLLM repository ==="
# Clone needed for DeepGEMM and EP kernels install scripts
cd $INSTALLATION_DIR
git clone https://github.com/vllm-project/vllm.git vllm
cd vllm
git checkout $VLLM_CHECKOUT_REF
echo "✓ vLLM repository cloned"

echo "\n=== Installing vLLM-Omni ==="
# Install omni BEFORE vLLM. Its transitive dependencies can otherwise upgrade the
# torch/transformers stack after vLLM is installed, which can leave vllm._C ABI-mismatched.
# vLLM should remain the final owner of the runtime stack in this environment.
if [ -n "$VLLM_OMNI_REF" ] && [ "$ARCH" = "amd64" ]; then
    # Try PyPI first, fall back to building from source
    if uv pip install ${VLLM_UV_ARGS} vllm-omni==${VLLM_OMNI_REF#v} 2>&1; then
        echo "✓ vLLM-Omni ${VLLM_OMNI_REF} installed from PyPI"
    else
        echo "⚠ PyPI install failed, building from source..."
        git clone --depth 1 --branch ${VLLM_OMNI_REF} https://github.com/vllm-project/vllm-omni.git $INSTALLATION_DIR/vllm-omni
        uv pip install ${VLLM_UV_ARGS} $INSTALLATION_DIR/vllm-omni
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

    verify_vllm_cuda_install() {
        # Run outside /opt/vllm so Python inspects the installed package, not the cloned source tree.
        (cd / && python3 - <<PY
from importlib import metadata
from pathlib import Path
import subprocess

import torch

expected = "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
if torch.version.cuda != expected:
    raise RuntimeError(
        f"PyTorch CUDA version {torch.version.cuda} does not match CUDA {expected}"
    )
print(f"✓ PyTorch CUDA version verified: {torch.version.cuda}")

vllm_version = metadata.version("vllm")
install_mode = "${VLLM_INSTALL_MODE}"
if install_mode == "wheel" and "${TORCH_BACKEND}" == "cu129" and not vllm_version.endswith("+cu129"):
    raise RuntimeError(f"Expected vLLM cu129 wheel, found vLLM {vllm_version}")

dist = metadata.distribution("vllm")
extension_paths = [
    Path(dist.locate_file(file))
    for file in (dist.files or [])
    if str(file).startswith("vllm/_C") and str(file).endswith(".so")
]
if not extension_paths:
    raise RuntimeError(f"Could not find vLLM extension libraries in vLLM {vllm_version}")

ldd_output = "\n".join(
    subprocess.check_output(["ldd", str(path)], text=True)
    for path in extension_paths
)
missing_cudart = [
    line.strip()
    for line in ldd_output.splitlines()
    if "libcudart.so" in line and "not found" in line
]
if missing_cudart:
    raise RuntimeError(
        "vLLM extension is linked against a missing CUDA runtime: "
        + "; ".join(missing_cudart)
    )
print(f"✓ vLLM extension CUDA runtime linkage verified: {vllm_version}")
PY
        )
    }

    if [ "$VLLM_INSTALL_MODE" = "wheel" ]; then
        # vLLM 0.20.0 switches the default PyPI CUDA wheel to CUDA 13.0.
        # Use the release wheel variant index for CUDA-specific vLLM binaries,
        # and ask uv for the matching torch backend for the PyTorch stack.
        echo "Installing vLLM ${VLLM_VER} (torch backend: $TORCH_BACKEND)..."
        uv pip install ${VLLM_UV_ARGS} "vllm[flashinfer,runai,otel]==${VLLM_VER}"
        echo "✓ vLLM ${VLLM_VER} installed from PyPI"
    else
        echo "Installing vLLM from source checkout ${VLLM_SOURCE_REF} (torch backend: $TORCH_BACKEND)..."
        # Source installs run without build isolation so we must satisfy key build requirements in-place.
        uv pip install \
            "setuptools>=77.0.3,<81.0.0" \
            "setuptools-scm>=8.0"
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}" \
               VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-$VLLM_SOURCE_REF}" \
               VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-$TORCH_BACKEND}"
        uv pip install --verbose --no-build-isolation \
            ".[flashinfer,runai,otel]" \
            ${TORCH_UV_ARGS}
        echo "✓ vLLM installed from source checkout ${VLLM_SOURCE_REF}"
    fi

    verify_vllm_cuda_install
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
