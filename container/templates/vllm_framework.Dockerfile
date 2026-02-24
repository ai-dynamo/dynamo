{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_framework.Dockerfile ===
########################################################
########## Framework Development Image ################
########################################################
#
# PURPOSE: Framework development and vLLM compilation
#
# This stage builds and compiles framework dependencies including:
# - vLLM inference engine with CUDA/XPU support
# - DeepGEMM and FlashInfer optimizations
# - All necessary build tools and compilation dependencies
# - Framework-level Python packages and extensions
#
# Use this stage when you need to:
# - Build vLLM from source with custom modifications
# - Develop or debug framework-level components
# - Create custom builds with specific optimization flags
#

# Use dynamo base image (see /container/Dockerfile for more details)
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

ARG PYTHON_VERSION
ARG DEVICE

# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Python runtime - CRITICAL for virtual environment to work
        python${PYTHON_VERSION}-dev \
        build-essential \
        # vLLM build dependencies
        cmake \
        ibverbs-providers \
        ibverbs-utils \
        libibumad-dev \
        libibverbs-dev \
        libnuma-dev \
        librdmacm-dev \
        rdma-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# if libmlx5.so not shipped with 24.04 rdma-core packaging, CMAKE will fail when looking for
# generic dev name .so so we symlink .s0.1 -> .so
RUN ln -sf /usr/lib/aarch64-linux-gnu/libmlx5.so.1 /usr/lib/aarch64-linux-gnu/libmlx5.so || true

# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

ARG ARCH
# Install vllm - keep this early in Dockerfile to avoid
# rebuilds from unrelated source code changes
ARG VLLM_REF
ARG VLLM_GIT_URL
ARG LMCACHE_REF
ARG VLLM_OMNI_REF

{% if device == "cuda" %}
ARG DEEPGEMM_REF
ARG FLASHINF_REF
ARG CUDA_VERSION
{% endif %}

ARG MAX_JOBS
ENV MAX_JOBS=$MAX_JOBS

{% if device == "cuda" %}
ENV CUDA_HOME=/usr/local/cuda
{% endif %}

{% if device == "xpu" %}
RUN wget --tries=3 --waitretry=5 https://raw.githubusercontent.com/zxue2/llm-scaler/3d9015157c6341ae5810eadfe3987cb9d1080a77/vllm/patches/ai-dynamo-xpu/patches/vllm-xpu-v0.14.0.patch -O /tmp/vllm-xpu.patch
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
{% endif %}

# Install VLLM and related dependencies
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    cp /tmp/deps/vllm/install_vllm.sh /tmp/install_vllm.sh && \
    chmod +x /tmp/install_vllm.sh && \
    /tmp/install_vllm.sh \
        --device $DEVICE \
        --vllm-ref $VLLM_REF \
        --max-jobs $MAX_JOBS \
        --arch $ARCH \
        --installation-dir /opt \
        ${LMCACHE_REF:+--lmcache-ref "$LMCACHE_REF"} \
        ${VLLM_OMNI_REF:+--vllm-omni-ref "$VLLM_OMNI_REF"} \
        ${DEEPGEMM_REF:+--deepgemm-ref "$DEEPGEMM_REF"} \
        ${FLASHINF_REF:+--flashinf-ref "$FLASHINF_REF"} \
        ${CUDA_VERSION:+--cuda-version "$CUDA_VERSION"}

{% if device == "cuda" %}
ENV LD_LIBRARY_PATH=\
/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:\
$LD_LIBRARY_PATH
{% endif %}
