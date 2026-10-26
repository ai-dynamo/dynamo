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

set -euo pipefail

PYTHON="${1:-python3}"
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERRIDES="$(mktemp)"
trap 'rm -f "${OVERRIDES}"' EXIT

# Keep the runtime's versions for the adapters' tested gRPC path. Riva 2.26's
# declared pins conflict; uv pip check still reports these metadata mismatches.
"${PYTHON}" - <<'PY' > "${OVERRIDES}"
from importlib.metadata import version

for package in ("protobuf", "websockets"):
    print(f"{package}=={version(package)}")
PY
uv pip install --python "${PYTHON}" --no-cache \
    --overrides "${OVERRIDES}" \
    --requirement "${EXAMPLE_DIR}/requirements.txt"
