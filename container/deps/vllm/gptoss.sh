#!/usr/bin/env bash
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

#################################
## Workarounds for gpt-oss
#################################
pip install tiktoken blobfile protobuf --break-system-packages

# fix for issue when using openai_harmony package which needs to download these files at runtime.
# on some closed networks this seems to cause failures. 
# https://github.com/openai/harmony/issues/46    
# Create encodings directory and download tiktoken encoding files
mkdir -p /opt/encodings
wget -O /opt/encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O /opt/encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
chmod -R 444 /opt/encodings
chown -R root:root /opt/encodings
