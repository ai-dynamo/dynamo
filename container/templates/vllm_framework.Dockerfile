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
# - vLLM inference engine with CUDA/XPU/CPU support
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

RUN apt clean && apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    curl ca-certificates zip unzip git lsb-release numactl wget vim

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

ARG TARGETARCH
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
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
{% endif %}

{% if device == "cpu" %}
## Use guidelines from https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#build-image-from-source
## to build a cross compiled target to support AVX512, AMX ISA's
## vllm-0.16 has a bug that handles non-AVX512 supported cases incorrectly
## -  https://github.com/vllm-project/vllm/issues/33991
## -  Build settings chosen to cross-compile with AVX512 support on amd64 only.

ENV VLLM_TARGET_DEVICE=cpu
ARG VLLM_CPU_X86=true              # Enable x86 ISA path in cmake (AVX2 + AVX512 + AMX)
{% endif %}

# Install VLLM and related dependencies
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    cp /tmp/deps/vllm/install_vllm.sh /tmp/install_vllm.sh && \
    chmod +x /tmp/install_vllm.sh && \
    if [ "$DEVICE" = "cpu" ] && [ "$TARGETARCH" = "amd64" ]; then \
        export VLLM_CPU_X86=${VLLM_CPU_X86}; \
    fi && \
    /tmp/install_vllm.sh \
        --device $DEVICE \
        --vllm-ref $VLLM_REF \
        --max-jobs $MAX_JOBS \
        --arch $TARGETARCH \
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
