#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script installs CUDA 13 during the KVBM wheel build.
# It compensates for the lack of a public CUDA manylinux image,
# while still allowing external users to build from source.

set -ex

ARCH_ALT="${ARCH_ALT:-$(uname -m)}"
case "$ARCH_ALT" in
  x86_64|amd64)
    ARCH_ALT="x86_64"
    CUDA_REPO_ARCH="x86_64"
    ;;
  aarch64|arm64)
    ARCH_ALT="aarch64"
    CUDA_REPO_ARCH="sbsa"
    ;;
  *)
    echo "Unsupported ARCH_ALT: $ARCH_ALT" >&2
    exit 1
    ;;
esac


NV_CUDA_CUDART_VERSION=13.0.88-1
NV_CUDA_LIB_VERSION=13.0.1-1
NV_NVTX_VERSION=13.0.85-1
NV_LIBNPP_VERSION=13.0.1.2-1
NV_LIBNPP_PACKAGE=libnpp-13-0-${NV_LIBNPP_VERSION}
NV_LIBCUBLAS_VERSION=13.0.2.14-1
NV_LIBNCCL_PACKAGE_NAME=libnccl
NV_LIBNCCL_PACKAGE_VERSION=2.28.3-1
NV_LIBNCCL_VERSION=2.28.3
NCCL_VERSION=2.28.3
NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda13.0

cat > "/etc/yum.repos.d/cuda.repo" <<EOF
[cuda]
name=cuda
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${CUDA_REPO_ARCH}
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA
EOF

NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${ARCH_ALT}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA
echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

yum upgrade -y && yum install -y \
    cuda-cudart-13-0-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-13-0 \
    cuda-toolkit-${NV_CUDA_LIB_VERSION} \
    cuda-nvcc-13-0-${NV_CUDA_CUDART_VERSION} \
    cuda-cudart-devel-13-0-${NV_CUDA_CUDART_VERSION} \
    cuda-libraries-13-0-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-13-0-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-13-0-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE}

# clean up
yum clean all
rm -rf /var/cache/yum/*
