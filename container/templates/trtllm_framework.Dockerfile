# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG BASE_IMAGE_TAG="25.10-cuda13.0-devel-ubuntu24.04"

ARG PYTORCH_BASE_IMAGE="nvcr.io/nvidia/pytorch"
ARG PYTORCH_BASE_IMAGE_TAG="25.10-py3"
ARG ENABLE_KVBM=false
ARG RUNTIME_IMAGE="nvcr.io/nvidia/cuda-dl-base"
ARG RUNTIME_IMAGE_TAG="25.10-cuda13.0-runtime-ubuntu24.04"

# TensorRT-LLM specific configuration
ARG HAS_TRTLLM_CONTEXT=0
ARG TENSORRTLLM_PIP_WHEEL="tensorrt-llm"
ARG TENSORRTLLM_INDEX_URL="https://pypi.nvidia.com/"
ARG GITHUB_TRTLLM_COMMIT

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
ARG ARCH=amd64
ARG ARCH_ALT=x86_64
# Python configuration
ARG PYTHON_VERSION=3.12

# Copy artifacts from NGC PyTorch image
FROM ${PYTORCH_BASE_IMAGE}:${PYTORCH_BASE_IMAGE_TAG} AS pytorch_base

##################################################
########## Framework Builder Stage ##############
##################################################
#
# PURPOSE: Build TensorRT-LLM with root privileges
#
# This stage handles TensorRT-LLM installation which requires:
# - Root access for apt operations (CUDA repos, TensorRT installation)
# - System-level modifications in install_tensorrt.sh
# - Virtual environment population with PyTorch and TensorRT-LLM
#
# The completed venv is then copied to runtime stage with dynamo ownership

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

ARG ARCH_ALT
ARG PYTHON_VERSION
ARG ENABLE_KVBM
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/${ARCH_ALT}-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install minimal dependencies needed for TensorRT-LLM installation
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        curl \
        git \
        git-lfs \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

# Copy pytorch installation from NGC PyTorch
ARG TORCH_VER=2.9.0a0+145a3a7bda.nv25.10
ARG TORCH_TENSORRT_VER=2.9.0a0
ARG TORCHVISION_VER=0.24.0a0+094e7af5
ARG JINJA2_VER=3.1.6
ARG SYMPY_VER=1.14.0
ARG FLASH_ATTN_VER=2.7.4.post1+25.10

COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch-${TORCH_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch-${TORCH_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchgen ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchgen
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision-${TORCHVISION_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision-${TORCHVISION_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torchvision.libs ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torchvision.libs
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/functorch ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/functorch
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/jinja2 ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/jinja2
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/jinja2-${JINJA2_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/jinja2-${JINJA2_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sympy ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/sympy
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sympy-${SYMPY_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/sympy-${SYMPY_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/flash_attn
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn-${FLASH_ATTN_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/flash_attn-${FLASH_ATTN_VER}.dist-info
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/flash_attn_2_cuda.cpython-*-*-linux-gnu.so ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt
COPY --from=pytorch_base /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt-${TORCH_TENSORRT_VER}.dist-info ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt-${TORCH_TENSORRT_VER}.dist-info

# Install TensorRT-LLM and related dependencies
ARG HAS_TRTLLM_CONTEXT
ARG TENSORRTLLM_PIP_WHEEL
ARG TENSORRTLLM_INDEX_URL
ARG GITHUB_TRTLLM_COMMIT

# Copy only wheel files and commit info from trtllm_wheel stage from build_context
COPY --from=trtllm_wheel /*.whl /trtllm_wheel/
COPY --from=trtllm_wheel /*.txt /trtllm_wheel/

RUN uv pip install --no-cache "cuda-python==13.0.2"

# Note: TensorRT needs to be uninstalled before installing the TRTLLM wheel
# because there might be mismatched versions of TensorRT between the NGC PyTorch
# and the TRTLLM wheel.
RUN [ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true && \
    # Clean up any existing conflicting CUDA repository configurations and GPG keys
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg && \
    rm -f /etc/apt/trusted.gpg.d/cuda*.gpg

RUN if [ "$HAS_TRTLLM_CONTEXT" = "1" ]; then \
        # Download and run install_tensorrt.sh from TensorRT-LLM GitHub before installing the wheel
        curl -fsSL --retry 5 --retry-delay 10 --max-time 1800 -o /tmp/install_tensorrt.sh "https://github.com/NVIDIA/TensorRT-LLM/raw/${GITHUB_TRTLLM_COMMIT}/docker/common/install_tensorrt.sh" && \
        # Modify the script to use virtual environment pip instead of system pip3
        sed -i 's/pip3 install/uv pip install/g' /tmp/install_tensorrt.sh && \
        bash /tmp/install_tensorrt.sh && \
        # Install from local wheel directory in build context
        WHEEL_FILE="$(find /trtllm_wheel -name "*.whl" | head -n 1)"; \
        if [ -n "$WHEEL_FILE" ]; then \
            uv pip install --no-cache "$WHEEL_FILE"; \
        else \
            echo "No wheel file found in /trtllm_wheel directory."; \
            exit 1; \
        fi; \
    else \
        # Download and run install_tensorrt.sh from TensorRT-LLM GitHub before installing the wheel
        TRTLLM_VERSION=$(echo "${TENSORRTLLM_PIP_WHEEL}" | sed -n 's/.*==\([0-9a-zA-Z\.\-]*\).*/\1/p') && \
        (curl -fsSL --retry 5 --retry-delay 10 --max-time 1800 -o /tmp/install_tensorrt.sh "https://github.com/NVIDIA/TensorRT-LLM/raw/v${TRTLLM_VERSION}/docker/common/install_tensorrt.sh" || \
         curl -fsSL --retry 5 --retry-delay 10 --max-time 1800 -o /tmp/install_tensorrt.sh "https://github.com/NVIDIA/TensorRT-LLM/raw/${GITHUB_TRTLLM_COMMIT}/docker/common/install_tensorrt.sh") && \
        # Modify the script to use virtual environment pip instead of system pip3
        sed -i 's/pip3 install/uv pip install/g' /tmp/install_tensorrt.sh && \
        bash /tmp/install_tensorrt.sh && \
        # Install TensorRT-LLM wheel from the provided index URL, allow dependencies from PyPI
        # TRTLLM 1.2.0rc2 has issues installing from pypi with uv, installing from direct wheel link works best
        # explicitly installing triton 3.5.0 as trtllm only lists triton as dependency on x64_64 for some reason
        export TENSORRTLLM_PIP_WHEEL="https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.2.0rc2-cp312-cp312-linux_${ARCH_ALT}.whl"; \
        uv pip install --no-cache --index-strategy=unsafe-best-match --extra-index-url "${TENSORRTLLM_INDEX_URL}" "${TENSORRTLLM_PIP_WHEEL}" triton==3.5.0; \
    fi
