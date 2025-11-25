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

# Install NIXL for TensorRT-LLM.
# This script is an adapted version of the NIXL install script from the TensorRT-LLM repository.
# The original script is located at:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/common/install_nixl.sh

set -ex

GITHUB_URL="https://github.com"

UCX_VERSION="v1.19.1"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"

NIXL_COMMIT="97c9b5b48e2ed3f1f2539c461c4971a7db8b1197"

UCX_REPO="https://github.com/openucx/ucx.git"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"




if [ ! -d ${UCX_INSTALL_PATH} ]; then
  git clone --depth 1 -b ${UCX_VERSION} ${UCX_REPO}
  cd ucx
  ./autogen.sh
  ./contrib/configure-release       \
    --prefix=${UCX_INSTALL_PATH}    \
    --enable-shared                 \
    --disable-static                \
    --disable-doxygen-doc           \
    --enable-optimizations          \
    --enable-cma                    \
    --enable-devel-headers          \
    --with-cuda=${CUDA_PATH}        \
    --with-verbs                    \
    --with-dm                       \
    --enable-mt
  make install -j$(nproc)
  cd ..
  rm -rf ucx  # Remove UCX source to save space
  echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
fi

ARCH_NAME="x86_64-linux-gnu"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
  ARCH_NAME="aarch64-linux-gnu"
  EXTRA_NIXL_ARGS="-Ddisable_gds_backend=true"
fi

if [ $ARCH_NAME != "x86_64-linux-gnu" ]; then
  echo "The NIXL backend is temporarily unavailable on the aarch64 platform. Exiting script."
  exit 0
fi

pip3 install --no-cache-dir meson ninja pybind11
git clone ${NIXL_REPO} nixl
cd nixl
git checkout ${NIXL_COMMIT}
meson setup builddir -Ducx_path=${UCX_INSTALL_PATH}  -Dstatic_plugins=UCX  -Dbuildtype=release ${EXTRA_NIXL_ARGS}
cd builddir && ninja install
cd ../..
rm -rf nixl*  # Remove NIXL source tree to save space

echo "export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/${ARCH_NAME}:/opt/nvidia/nvda_nixl/lib64:\$LD_LIBRARY_PATH" >> "${ENV}"
