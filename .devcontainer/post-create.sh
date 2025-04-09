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

# Function to retry a command up to a specified number of times

retry() {
    # retries for connectivity issues in pip install
    local retries=3
    local count=0
    until "$@"; do
        exit_code=$?
        wait_time=$((2 ** count))
        echo "Command failed with exit code $exit_code. Retrying in $wait_time seconds..."
        sleep $wait_time
        count=$((count + 1))
        if [ $count -ge $retries ]; then
            echo "Command failed after $retries attempts."
            return $exit_code
        fi
    done
    return 0
}

set -xe
export CARGO_BUILD_JOBS=32
export CARGO_TARGET_DIR=$HOME/dynamo/.build/target

cd $HOME/dynamo
# Treated as a local cache for the build
mkdir -p $HOME/dynamo/.build

# build project, it will be saved at $HOME/dynamo/.build/target
cargo build --profile dev --locked --features mistralrs,sglang,vllm,python
cargo doc --no-deps

# create symlinks for the binaries in the deploy directory
mkdir -p $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
ln -sf $HOME/dynamo/.build/target/debug/dynamo-run $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/dynamo-run
ln -sf $HOME/dynamo/.build/target/debug/http $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/http
ln -sf $HOME/dynamo/.build/target/debug/llmctl $HOME/dynamo/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin/llmctl

# install the python bindings in editable mode
retry bash -c 'DYNAMO_BIN_PATH=$HOME/dynamo/.build/target/debug uv pip install -e .'
cd $HOME/dynamo/lib/bindings/python && retry uv pip install -e .

# source the venv and set the VLLM_KV_CAPI_PATH in bashrc
echo "source /opt/dynamo/venv/bin/activate" >> ~/.bashrc
echo "export VLLM_KV_CAPI_PATH=$HOME/dynamo/.build/target/debug/libdynamo_llm_capi.so" >> ~/.bashrc