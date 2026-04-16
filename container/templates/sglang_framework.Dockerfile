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
# The intel/deep-learning-essentials base image already includes oneAPI + Python
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        intel-ocloc \
        libsqlite3-dev \
        python${PYTHON_VERSION}-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy uv from dynamo_base
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create virtual environment
RUN uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Install PyTorch XPU packages (matching upstream xpu.Dockerfile)
WORKDIR /sgl-workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    source ${VIRTUAL_ENV}/bin/activate && \
    uv pip install \
        torch==2.11.0+xpu \
        torchao \
        torchvision \
        torchaudio==2.11.0+xpu \
        triton-xpu==3.7.0 \
        --index-url https://download.pytorch.org/whl/xpu

# Install sgl-kernel-xpu first — it uses CMake find_package(Torch), so torch must
# be visible at build time.  We use --no-build-isolation, which means the build-
# system deps (scikit-build-core, cmake, ninja, setuptools) must live in the venv.
# IMPORTANT: source setvars.sh so CMake picks up icpx (DPCPP) compiler for SYCL kernels.
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    source /opt/intel/oneapi/setvars.sh --force && \
    source ${VIRTUAL_ENV}/bin/activate && \
    uv pip install scikit-build-core cmake ninja setuptools && \
    uv pip install "sgl-kernel @ git+${SGLANG_KERNEL_GIT_URL}@${SGLANG_KERNEL_REF}" --no-build-isolation

# Clone SGLang and install for XPU (sgl-kernel is already satisfied from above)
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone ${SGLANG_GIT_URL} sglang && \
    cd sglang && \
    git checkout ${SGLANG_REF} && \
    cd python && \
    cp pyproject_xpu.toml pyproject.toml && \
    uv pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu . && \
    uv pip install xgrammar --no-deps && \
    uv pip install msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser einops tabulate

# Source oneAPI environment in bashrc for interactive shells
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /etc/bash.bashrc

ENV SGLANG_FORCE_SHUTDOWN=1

{% endif %}
# === END templates/sglang_framework.Dockerfile ===
