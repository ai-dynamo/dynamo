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

# Ensure we're not running as root
if [ "$(id -u)" -eq 0 ]; then
    echo "❌ ERROR: This script should not be run as root!"
    echo "The script should run as the 'ubuntu' user, not root."
    echo "Current user: $(whoami) (UID: $(id -u))"
    exit 1
fi

# Verify we're running as the expected user
if [ "$(whoami)" != "ubuntu" ]; then
    echo "⚠️  WARNING: Expected to run as 'ubuntu' user, but running as '$(whoami)'"
    echo "This might cause permission issues."
fi

echo "✅ Running post-create script as user: $(whoami) (UID: $(id -u))"

trap 'echo "❌ ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "⚠️ This was unexpected and setup was not completed. Can try to resolve yourself and then manually run the rest of the commands in this file or file a bug."' ERR

retry() {
    # retries for connectivity issues in installs
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

set -eux

# Changing permission to match local user since volume mounts default to root ownership
# Note: sudo is used here because the volume mount may have root ownership
mkdir -p $HOME/.cache
sudo chown -R ubuntu:ubuntu $HOME/.cache $HOME/dynamo

# Pre-commit hooks
cd $HOME/dynamo && pre-commit install && retry pre-commit install-hooks
pre-commit run --all-files || true # don't fail the build if pre-commit hooks fail

# Set build directory
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/dynamo/.build/target}
mkdir -p $CARGO_TARGET_DIR

# Build project, with `dev` profile it will be saved at $CARGO_TARGET_DIR/debug
cargo build --locked --profile dev --features mistralrs
cargo doc --no-deps

# install the python bindings
cd $HOME/dynamo/lib/bindings/python && retry maturin develop

# installs overall python packages, grabs binaries from .build/target/debug
cd $HOME/dynamo && retry env DYNAMO_BIN_PATH=$CARGO_TARGET_DIR/debug uv pip install -e .

if [ -z "${PYTHONPATH+x}" ]; then
    export PYTHONPATH=/home/ubuntu/dynamo/components/planner/src
else
    export PYTHONPATH=/home/ubuntu/dynamo/components/planner/src:$PYTHONPATH
fi

# TODO: Deprecated except vLLM v0
if ! grep -q "export VLLM_KV_CAPI_PATH=" ~/.bashrc; then
    echo "export VLLM_KV_CAPI_PATH=$CARGO_TARGET_DIR/debug/libdynamo_llm_capi.so" >> ~/.bashrc
fi

if ! grep -q "export GPG_TTY=" ~/.bashrc; then
    echo "export GPG_TTY=$(tty)" >> ~/.bashrc
fi

{ set +x; } 2>/dev/null

echo ""
echo "✅ SUCCESS: Built cargo project, installed Python bindings, configured pre-commit hooks"
echo ""
echo "Example commands:"
echo "  cargo build --locked --profile dev         # Build Rust project in $CARGO_TARGET_DIR"
echo "  cd lib/bindings/python && maturin develop  # Update Python bindings"
echo "  cargo fmt && cargo clippy                  # Format and lint code before committing"
