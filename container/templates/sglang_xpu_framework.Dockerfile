{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_xpu_framework.Dockerfile ===

##################################
#### SGLang XPU Framework ########
##################################
#
# PURPOSE: Build SGLang from source for Intel XPU (GPU) environments.
#
# This stage follows the pattern from sgl-project/sglang/docker/xpu.Dockerfile.
# It builds SGLang with XPU PyTorch (Intel GPU via oneAPI/Level Zero).
#
# The resulting image is used as the base for sglang_runtime when device=xpu.
#

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

ARG TARGETARCH
ARG PYTHON_VERSION
ARG SGLANG_REF
ARG SGLANG_GIT_URL
ARG SGLANG_KERNEL_GIT_URL
ARG SGLANG_KERNEL_REF

SHELL ["/bin/bash", "-c"]

# Install additional system dependencies for XPU build
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# Pin the Level-Zero UMD + IGC, matching upstream sgl-kernel-xpu's
# Dockerfile.xpu_kernel for the v0.2.0 tag. These must stay in lockstep with
# the host xe KMD: sgl-kernel-xpu#296 saw a mismatched UMD fault libze on
# Battlemage. The compute-runtime release also carries libze-intel-gpu1's
# matching intel-ocloc / intel-opencl-icd, so all three move together.
#
# Only the GPU driver (UMD) is pinned here. The Level Zero *loader* comes from
# the base image, which ships libze1 + libze-dev 1.27.0 from Intel's
# kobuk-team PPA. The 2025.3 base did not, which is why a standalone
# level-zero deb used to be fetched; installing it now fails, because it
# and libze1 both own /usr/lib/x86_64-linux-gnu/libze_loader.so.1. Upstream
# v0.5.19 relies on the PPA loader for the same reason.
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG IGC_VERSION=2.34.4+21428
ARG GMM_VERSION=22.10.0

RUN mkdir -p /tmp/neo && cd /tmp/neo && \
    WGET="wget -q --tries=5 --waitretry=5 --retry-connrefused --retry-on-http-error=429,500,502,503,504" && \
    IGC_URL="https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION%%+*}" && \
    CR_URL="https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}" && \
    $WGET ${IGC_URL}/intel-igc-core-2_${IGC_VERSION}_amd64.deb && \
    $WGET ${IGC_URL}/intel-igc-opencl-2_${IGC_VERSION}_amd64.deb && \
    $WGET ${CR_URL}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    $WGET ${CR_URL}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    $WGET ${CR_URL}/libigdgmm12_${GMM_VERSION}_amd64.deb && \
    $WGET ${CR_URL}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades ./*.deb && \
    rm -rf /var/lib/apt/lists/* && \
    cd / && rm -rf /tmp/neo && \
    apt-mark hold libze-intel-gpu1 intel-opencl-icd intel-ocloc libigdgmm12 \
        intel-igc-core-2 intel-igc-opencl-2

# Install Miniforge (conda) — follows upstream sgl-project/sglang/docker/xpu.Dockerfile pattern.
# Conda provides correct library linkage with the base image's oneAPI/Level Zero stack.
ENV CONDA_DIR=/opt/miniforge3
RUN curl -fsSL --retry 5 --retry-delay 5 --retry-connrefused -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/download/25.1.1-0/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda create -y -n sglang python=${PYTHON_VERSION} && \
    ${CONDA_DIR}/bin/conda run -n sglang conda install -y pip

ENV VIRTUAL_ENV="${CONDA_DIR}/envs/sglang" \
    PATH="${CONDA_DIR}/envs/sglang/bin:${CONDA_DIR}/bin:${PATH}" \
    CONDA_DEFAULT_ENV=sglang

# Install PyTorch XPU packages. Versions track sglang v0.5.19's
# python/pyproject_xpu.toml exactly — pinning them here (rather than letting the
# sglang install below resolve them) keeps the sgl-kernel-xpu build, which
# compiles against this torch, on the same ABI. torch 2.13.0+xpu requires
# triton-xpu 3.7.2, so it is no longer pinned separately below.
# torchao is not a v0.5.19 XPU dependency (nothing under sglang/ imports it;
# it appears only in check_env's optional report), so it is dropped rather than
# resolved to a build that would pull a conflicting torch.
WORKDIR /sgl-workspace
RUN pip3 install \
        torch==2.13.0+xpu \
        torchvision==0.28.0+xpu \
        torchaudio==2.11.0+xpu \
        --index-url https://download.pytorch.org/whl/xpu

# Install sgl-kernel-xpu — needs icpx (DPCPP) for SYCL kernels.
# Uses --no-build-isolation so build deps must be pre-installed.
#
# DPCPP_SYCL_TARGET is set explicitly: v0.2.0 auto-detects the AOT target by
# running a Level Zero probe against a live GPU, which is not available inside
# `docker build`. Without it the probe fails and CMake silently falls back to
# `bmg`, so state the target rather than depend on that fallback. bmg = Xe2
# (Arc A/B-series, Arc Pro B60); use cri for Xe3P.
ARG DPCPP_SYCL_TARGET=bmg
RUN source /opt/intel/oneapi/setvars.sh --force && \
    pip3 install scikit-build-core cmake ninja setuptools && \
    pip3 install "sglang-kernel-xpu @ git+${SGLANG_KERNEL_GIT_URL}@${SGLANG_KERNEL_REF}" \
        --no-build-isolation \
        --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET}

# Clone SGLang and install for XPU.
#
# The kernel requirement is dropped from pyproject.toml before installing.
# pyproject_xpu.toml at v0.5.19 declares
# `sgl-kernel @ git+https://github.com/sgl-project/sgl-kernel-xpu.git`, i.e. the
# kernel repo's DEFAULT BRANCH under its OLD distribution name. That is stale two
# ways: the repo renamed the distribution to `sglang-kernel-xpu` at v0.2.0, so pip
# aborts on the name mismatch ("has inconsistent name"); and even resolved it
# would re-clone main and recompile the SYCL kernels, discarding the pinned
# ${SGLANG_KERNEL_REF} build above and adding a long build for an unpinned ref.
# Upstream main has since repointed this at a released wheel. Removing the line
# leaves the kernel install above as the single source of the pin.
#
# The grep guard makes the edit fail loudly rather than silently no-op if a
# future ${SGLANG_REF} renames or drops the requirement.
RUN git clone ${SGLANG_GIT_URL} sglang && \
    cd sglang && \
    git checkout ${SGLANG_REF} && \
    cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    grep -q '^\s*"sgl-kernel @ git+' pyproject.toml && \
    sed -i '/^\s*"sgl-kernel @ git+/d' pyproject.toml && \
    pip3 install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu ".[diffusion]" && \
    pip3 install "xgrammar==0.1.33" --no-deps && \
    pip3 install msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate ftfy

# Multimodal + accelerate runtime deps that pyproject_xpu.toml does NOT pull in
# via the `[diffusion]` extra above:
#   - decord: dropped from pyproject_xpu.toml entirely (CUDA pyproject.toml
#     ships decord2 by default for multimodal video decode).
#   - accelerate: only declared in pyproject_xpu.toml's `[test]` extra, but
#     diffusers needs it at runtime for enable_model_cpu_offload.
# Use --no-deps so the resolver doesn't pull CUDA-bound transitive packages.
RUN pip3 install --no-deps decord accelerate

# pyproject_xpu.toml's [diffusion] extra pulls opencv-python (with X11/libGL
# deps), but the container image has no libGL.so.1, so `import cv2` fails.
# Swap to opencv-python-headless (same version) — matches the CUDA pyproject
# default and avoids dragging X11 libs into the image.
RUN pip3 uninstall -y opencv-python && \
    pip3 install --no-deps "opencv-python-headless==4.10.0.84"

# Source conda + oneAPI environment in bashrc for interactive shells
RUN echo ". ${CONDA_DIR}/bin/activate sglang" >> /etc/bash.bashrc && \
    echo "source /opt/intel/oneapi/setvars.sh --force" >> /etc/bash.bashrc

ENV SGLANG_FORCE_SHUTDOWN=1

# === END templates/sglang_xpu_framework.Dockerfile ===
