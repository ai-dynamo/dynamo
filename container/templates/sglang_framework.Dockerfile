{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_framework.Dockerfile ===
##################################
#### SGLang CPU Framework ########
##################################
#
# PURPOSE: Build SGLang from source for CPU-only environments.
#
# This stage follows the pattern from sgl-project/sglang/docker/xeon.Dockerfile.
# It builds SGLang with CPU-only PyTorch (no CUDA/GPU dependencies).
#
# The resulting image is used as the base for sglang_runtime when device=cpu.
#

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

ARG TARGETARCH
ARG PYTHON_VERSION
ARG SGLANG_REF
ARG SGLANG_GIT_URL

SHELL ["/bin/bash", "-c"]

# Install system dependencies following sgl-project/sglang xeon.Dockerfile
RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        gcc \
        g++ \
        make \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        libsqlite3-dev \
        google-perftools \
        libtbb-dev \
        libnuma-dev \
        numactl && \
    rm -rf /var/lib/apt/lists/*

# Copy uv from dynamo_base
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create virtual environment
RUN uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Clone SGLang and build for CPU using pyproject_cpu.toml (same as xeon.Dockerfile)
WORKDIR /sgl-workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone ${SGLANG_GIT_URL} sglang && \
    cd sglang && \
    git checkout ${SGLANG_REF} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu . && \
    cd ../sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu .

ENV SGLANG_USE_CPU_ENGINE=1
ENV SGLANG_FORCE_SHUTDOWN=1

# === END templates/sglang_framework.Dockerfile ===
