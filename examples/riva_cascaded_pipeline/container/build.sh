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

# Build a Dynamo image with the RIVA NIM Python client layered on top, for the
# cascaded voice pipeline example.
#
#   BASE_IMAGE  Dynamo image to layer on (default: dynamo:latest-vllm-runtime)
#   TAG         Output image tag        (default: dynamo-riva-custom:latest)

set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-dynamo:latest-vllm-runtime}"
TAG="${TAG:-dynamo-riva-custom:latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Building ${TAG} from base ${BASE_IMAGE}"
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${EXAMPLE_DIR}"

echo "Built ${TAG}"
