#!/usr/bin/env bash

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

# Backwards-compatible shim. The agentgateway installer now lives next to its
# gateway documentation in deploy/inference-gateway/gateways/agentgateway.
# Prefer invoking that script directly; this file forwards to it so older
# callers (CI workflows, docs) keep working.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET="${SCRIPT_DIR}/../gateways/agentgateway/install.sh"

echo "scripts/install_gaie_crd_agentgateway.sh is deprecated; forwarding to gateways/agentgateway/install.sh." >&2
exec "${TARGET}" "$@"
