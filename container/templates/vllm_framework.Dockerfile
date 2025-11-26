# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG VLLM_BASE_IMAGE={{ context.vllm[platform].base_image }}
# TODO OPS-612: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
ARG VLLM_BASE_IMAGE_TAG={{ context.vllm[platform].base_image_tag }}
ARG ENABLE_KVBM={{ context.dynamo_base.enable_kvbm }}
ARG RUNTIME_IMAGE={{ context.vllm[platform].runtime_image }}
ARG RUNTIME_IMAGE_TAG={{ context.vllm[platform].runtime_image_tag }}
ARG CUDA_VERSION={{ context.vllm[platform].cuda_version }}

# Make sure to update the dependency version in pyproject.toml when updating this
ARG VLLM_REF={{ context.vllm.ref }}
# FlashInfer only respected when building vLLM from source, ie when VLLM_REF does not start with 'v' or for arm64 builds
ARG FLASHINF_REF={{ context.vllm.flashinf_ref }}
ARG TORCH_BACKEND={{ context.vllm[platform].torch_backend }}

# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""

# sccache configuration - inherit from base build
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# Define general architecture ARGs for supporting both x86 and aarch64 builds.
#   ARCH: Used for package suffixes (e.g., amd64, arm64)
#   ARCH_ALT: Used for Rust targets, manylinux suffix (e.g., x86_64, aarch64)
#
# Default values are for x86/amd64:
#   --build-arg ARCH=amd64 --build-arg ARCH_ALT=x86_64
#
# For arm64/aarch64, build with:
#   --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64
#
# NOTE: There isn't an easy way to define one of these values based on the other value
# without adding if statements everywhere, so just define both as ARGs for now.
ARG ARCH={{ platform }}
ARG ARCH_ALT={{ "x86_64" if platform == "amd64" else "aarch64" }}
# Python configuration
ARG PYTHON_VERSION={{ context.dynamo_base.python_version }}

########################################################
########## Framework Development Image ################
########################################################
#
# PURPOSE: Framework development and vLLM compilation
#
# This stage builds and compiles framework dependencies including:
# - vLLM inference engine with CUDA support
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
FROM ${VLLM_BASE_IMAGE}:${VLLM_BASE_IMAGE_TAG} AS framework

ARG PYTHON_VERSION

RUN apt-get update -y \
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

### VIRTUAL ENVIRONMENT SETUP ###

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

ARG ARCH
# Install vllm - keep this early in Dockerfile to avoid
# rebuilds from unrelated source code changes
ARG VLLM_REF
ARG VLLM_GIT_URL
ARG DEEPGEMM_REF
ARG FLASHINF_REF
ARG TORCH_BACKEND
ARG CUDA_VERSION

ARG MAX_JOBS=16
ENV MAX_JOBS=$MAX_JOBS
ENV CUDA_HOME=/usr/local/cuda

# Install sccache if requested
COPY container/use-sccache.sh /tmp/use-sccache.sh
# Install sccache if requested
ARG USE_SCCACHE
ARG ARCH_ALT
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION

ENV ARCH_ALT=${ARCH_ALT}
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

# Set environment variables - they'll be empty strings if USE_SCCACHE=false
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}} \
    CMAKE_C_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CXX_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CUDA_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache}
# Install VLLM and related dependencies
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
        cp /tmp/deps/vllm/install_vllm.sh /tmp/install_vllm.sh && \
        chmod +x /tmp/install_vllm.sh && \
        /tmp/install_vllm.sh --editable --vllm-ref $VLLM_REF --max-jobs $MAX_JOBS --arch $ARCH --installation-dir /opt ${DEEPGEMM_REF:+--deepgemm-ref "$DEEPGEMM_REF"} ${FLASHINF_REF:+--flashinf-ref "$FLASHINF_REF"} --torch-backend $TORCH_BACKEND --cuda-version $CUDA_VERSION && \
        /tmp/use-sccache.sh show-stats "vLLM";

ENV LD_LIBRARY_PATH=\
/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:\
$LD_LIBRARY_PATH
