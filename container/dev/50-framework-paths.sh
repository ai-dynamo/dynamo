#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Framework-specific environment variables and paths
# Only add paths that exist to avoid cluttering environment

# TensorRT-LLM specific variables
if [ -d /usr/local/tensorrt/targets ]; then
    export TENSORRT_LIB_DIR=/usr/local/tensorrt/targets/$(uname -m)-linux-gnu/lib
    [ -d "$TENSORRT_LIB_DIR" ] && export LD_LIBRARY_PATH="${TENSORRT_LIB_DIR}:${LD_LIBRARY_PATH}"
fi

if [ -d /opt/hpcx/ompi ]; then
    export OPAL_PREFIX=/opt/hpcx/ompi
    export OMPI_MCA_coll_ucc_enable=0
    export PATH="/opt/hpcx/ompi/bin:${PATH}"
    export LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}"
fi

[ -d /opt/hpcx/ucc/lib ] && export LD_LIBRARY_PATH="/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"
[ -f /etc/shinit_v2 ] && export ENV="${ENV:-/etc/shinit_v2}"
[ -d /usr/local/ucx/bin ] && export PATH="/usr/local/ucx/bin:${PATH}"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:${PATH}"
[ -d /usr/local/cuda/nvvm/bin ] && export PATH="/usr/local/cuda/nvvm/bin:${PATH}"

# vLLM nvshmem
[ -d /opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib ] && \
    export LD_LIBRARY_PATH="/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:${LD_LIBRARY_PATH}"

# System nvshmem (TRT-LLM)
ARCH_ALT=$(uname -m | sed 's/aarch64/aarch64/;s/x86_64/x86_64/')
[ -d "/usr/lib/${ARCH_ALT}-linux-gnu/nvshmem/13" ] && \
    export LD_LIBRARY_PATH="/usr/lib/${ARCH_ALT}-linux-gnu/nvshmem/13:${LD_LIBRARY_PATH}"

# PyTorch libraries (TRT-LLM)
# PYTHON_VERSION should be set via ENV in container; fail early if missing
if [ -z "${PYTHON_VERSION}" ]; then
    echo "WARNING: PYTHON_VERSION not set, defaulting to 3.12" >&2
    PYTHON_VERSION=3.12
fi
[ -d "/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib" ] && \
    export LD_LIBRARY_PATH="/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${LD_LIBRARY_PATH}"
[ -d "/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt/lib" ] && \
    export LD_LIBRARY_PATH="/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH}"
