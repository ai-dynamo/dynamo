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

# Build the CPU-only Dynamo-to-Riva adapter image for the cascaded voice
# pipeline example. The published Dynamo frontend image supplies the runtime;
# this layer adds only the Riva client and adapter code.
#
#   DYNAMO_FRONTEND_IMAGE     Published Dynamo frontend base image
#   CUSTOM_RIVA_ADAPTER_IMAGE Output adapter image tag

set -euo pipefail

: "${DYNAMO_FRONTEND_IMAGE:?Set DYNAMO_FRONTEND_IMAGE to a published Dynamo frontend image}"
: "${CUSTOM_RIVA_ADAPTER_IMAGE:?Set CUSTOM_RIVA_ADAPTER_IMAGE to the output adapter image tag}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Building ${CUSTOM_RIVA_ADAPTER_IMAGE} from ${DYNAMO_FRONTEND_IMAGE}"
docker build \
  --build-arg "BASE_IMAGE=${DYNAMO_FRONTEND_IMAGE}" \
  -t "${CUSTOM_RIVA_ADAPTER_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${EXAMPLE_DIR}"

echo "Built ${CUSTOM_RIVA_ADAPTER_IMAGE}"
