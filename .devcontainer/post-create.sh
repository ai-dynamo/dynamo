#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -xe

cd $HOME/dynamo

export CARGO_BUILD_JOBS=32
export CARGO_TARGET_DIR=$HOME/dynamo/target

cargo build --release --features vllm,python

mkdir -p $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
ln -sf $HOME/dynamo/target/release/http $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/http
ln -sf $HOME/dynamo/target/release/llmctl $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/llmctl

sudo chmod -R a+rw /opt/dynamo/venv

uv pip install -e .

echo "source /opt/dynamo/venv/bin/activate" >> ~/.bashrc
echo "export VLLM_KV_CAPI_PATH=$HOME/dynamo/target/release/libdynamo_llm_capi.so" >> ~/.bashrc

source ~/.bashrc