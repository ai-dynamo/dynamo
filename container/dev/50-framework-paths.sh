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

# /opt/dynamo/mpi is the Open MPI that trtllm_runtime.Dockerfile selects per
# architecture, and its ENV already points there. Prefer it, so that a login
# shell does not switch back to the base image's default at /opt/hpcx/ompi.
if [ -d /opt/dynamo/mpi ]; then
    _mpi_prefix=/opt/dynamo/mpi
elif [ -d /opt/hpcx/ompi ]; then
    _mpi_prefix=/opt/hpcx/ompi
else
    _mpi_prefix=
fi
if [ -n "${_mpi_prefix}" ]; then
    export OPAL_PREFIX="${_mpi_prefix}"
    export OMPI_MCA_coll_ucc_enable=0
    export PATH="${_mpi_prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_mpi_prefix}/lib:${LD_LIBRARY_PATH}"
fi
unset _mpi_prefix

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
