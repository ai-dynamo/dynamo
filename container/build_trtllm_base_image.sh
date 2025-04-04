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

# Build the TRT-LLM base image.

# This script builds the TRT-LLM base image for Dynamo with TensorRT-LLM.
TRTLLM_COMMIT=9b931c0f6

while getopts "c:" opt; do
  case ${opt} in
    c) TRTLLM_COMMIT=$OPTARG ;;
    *) echo "Invalid option" ;;
  esac
done

(cd /tmp && \
# Clone the TensorRT-LLM repository.
if [ ! -d "TensorRT-LLM" ]; then
  git clone https://github.com/NVIDIA/TensorRT-LLM.git
fi

cd TensorRT-LLM

# Checkout the specified commit.
git checkout $TRTLLM_COMMIT

# Update the submodules.
git submodule update --init --recursive
git lfs pull

# Build the TRT-LLM base image.
make -C docker release_build)
