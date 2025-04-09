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
# Setup GPG environment

set -ex

echo "Setting up GPG environment"

# Ensure clean GPG directory with correct permissions
rm -rf ~/.gnupg
mkdir -p ~/.gnupg
chmod 700 ~/.gnupg

# Copy GPG files from the read-only mount
cp /tmp/gpg-import/pubring.kbx ~/.gnupg/
cp /tmp/gpg-import/trustdb.gpg ~/.gnupg/
mkdir -p ~/.gnupg/private-keys-v1.d
cp -r /tmp/gpg-import/private-keys-v1.d/* ~/.gnupg/private-keys-v1.d/

# Set proper permissions
find ~/.gnupg -type f -exec chmod 600 {} \;
find ~/.gnupg -type d -exec chmod 700 {} \;

# Start GPG agent
export GPG_TTY=$(tty)
gpg-agent --daemon

echo "GPG environment setup complete"