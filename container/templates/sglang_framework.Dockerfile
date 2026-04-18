{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_framework.Dockerfile ===

{% if device == "xpu" %}
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
        libsqlite3-dev \
        intel-ocloc && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda) — follows upstream sgl-project/sglang/docker/xpu.Dockerfile pattern.
# Conda provides correct library linkage with the base image's oneAPI/Level Zero stack.
ENV CONDA_DIR=/opt/miniforge3
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/download/25.1.1-0/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda create -y -n sglang python=${PYTHON_VERSION} && \
    ${CONDA_DIR}/bin/conda run -n sglang conda install -y pip

ENV VIRTUAL_ENV="${CONDA_DIR}/envs/sglang" \
    PATH="${CONDA_DIR}/envs/sglang/bin:${CONDA_DIR}/bin:${PATH}" \
    CONDA_DEFAULT_ENV=sglang

# Install PyTorch XPU packages (matching upstream xpu.Dockerfile)
WORKDIR /sgl-workspace
RUN pip3 install \
        torch==2.11.0+xpu \
        torchao \
        torchvision \
        torchaudio==2.11.0+xpu \
        --index-url https://download.pytorch.org/whl/xpu

RUN pip3 install triton-xpu==3.7.0

# Install sgl-kernel-xpu — needs icpx (DPCPP) for SYCL kernels.
# Uses --no-build-isolation so build deps must be pre-installed.
RUN source /opt/intel/oneapi/setvars.sh --force && \
    pip3 install scikit-build-core cmake ninja setuptools && \
    pip3 install "sgl-kernel @ git+${SGLANG_KERNEL_GIT_URL}@${SGLANG_KERNEL_REF}" --no-build-isolation

# Clone SGLang and install for XPU (sgl-kernel is already satisfied from above)
RUN git clone ${SGLANG_GIT_URL} sglang && \
    cd sglang && \
    git checkout ${SGLANG_REF} && \
    cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    pip3 install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu . && \
    pip3 install xgrammar --no-deps && \
    pip3 install msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate

# Source conda + oneAPI environment in bashrc for interactive shells
RUN echo ". ${CONDA_DIR}/bin/activate sglang" >> /etc/bash.bashrc && \
    echo "source /opt/intel/oneapi/setvars.sh --force" >> /etc/bash.bashrc

ENV SGLANG_FORCE_SHUTDOWN=1

{% endif %}
# === END templates/sglang_framework.Dockerfile ===
