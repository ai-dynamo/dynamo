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

set -e

# Ensure Rust toolchain is in PATH
export PATH="/usr/local/cargo/bin:${PATH}"

# Create and activate Python virtual environment
python3 -m venv venv
. ./venv/bin/activate

# Verify Rust installation
rustc --version
cargo --version

# Build Rust components first
cargo build --release

# Install development tools with caching
uv pip install --upgrade pip setuptools wheel
uv pip install --cache-dir /tmp/pip-cache pytest isort mypy pylint pre-commit

# Install Dynamo with all dependencies
uv pip install --cache-dir /tmp/pip-cache -e .[all]

echo "Development environment setup complete!"