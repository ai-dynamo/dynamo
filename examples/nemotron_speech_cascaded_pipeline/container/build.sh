#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build the CPU-only Dynamo adapter image for Speech NIMs for the cascaded voice
# pipeline example. The Rust worker and its Dynamo runtime are compiled from
# this checkout; the final image requires neither Python nor a GPU runtime.
#
#   CUSTOM_SPEECH_ADAPTER_IMAGE Output adapter image tag

set -euo pipefail

: "${CUSTOM_SPEECH_ADAPTER_IMAGE:?Set CUSTOM_SPEECH_ADAPTER_IMAGE to the output adapter image tag}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Building ${CUSTOM_SPEECH_ADAPTER_IMAGE} from the local Rust sources"
docker build \
  -t "${CUSTOM_SPEECH_ADAPTER_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${REPO_ROOT}"

echo "Built ${CUSTOM_SPEECH_ADAPTER_IMAGE}"
