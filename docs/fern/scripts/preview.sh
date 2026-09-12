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

# Usage: bash docs/fern/scripts/preview.sh [port]
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREVIEW_DIR="${DOCS_DIR}/fern/.preview"
PREVIEW_PORT="${1:-3000}"
PREVIEW_PYTHON_BOOTSTRAP="${PYTHON:-python3}"
for dependency in "${PREVIEW_PYTHON_BOOTSTRAP}" node npm; do
  command -v "${dependency}" >/dev/null || {
    echo "Missing ${dependency}; install Python 3.11+ and Node.js 22+." >&2
    exit 1
  }
done
"${PREVIEW_PYTHON_BOOTSTRAP}" - "${PREVIEW_PORT}" <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required.")
try:
    port = int(sys.argv[1])
except ValueError:
    raise SystemExit("Port must be an integer between 1024 and 65534.")
if not 1024 <= port <= 65534:
    raise SystemExit("Port must be between 1024 and 65534; the next port serves the backend.")
PY
PREVIEW_BACKEND_PORT="$((PREVIEW_PORT + 1))"
mkdir -p "${PREVIEW_DIR}"
if [[ ! -x "${PREVIEW_DIR}/venv/bin/python" ]]; then
  "${PREVIEW_PYTHON_BOOTSTRAP}" -m venv "${PREVIEW_DIR}/venv"
fi
PREVIEW_PYTHON="${PREVIEW_DIR}/venv/bin/python"
if ! "${PREVIEW_PYTHON}" -c 'from importlib.metadata import version; assert version("griffe") == "2.1.0"' 2>/dev/null; then
  "${PREVIEW_PYTHON}" -m pip install 'griffe==2.1.0'
fi

# These generated pages and assets are gitignored, as in the publish pipeline.
"${PREVIEW_PYTHON}" "${DOCS_DIR}/fern/scripts/gen_python_api.py"
"${PREVIEW_PYTHON}" "${DOCS_DIR}/fern/scripts/gen_rust_api.py"
"${PREVIEW_PYTHON}" "${DOCS_DIR}/fern/scripts/gen_llms_tables.py" --assets-only
FERN_VERSION="$("${PREVIEW_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "${DOCS_DIR}/fern/fern.config.json")"

cd "${DOCS_DIR}"
echo "Preview: http://localhost:${PREVIEW_PORT}/dynamo/dev/recipes/ax-k2"
echo "For remote access, forward ports ${PREVIEW_PORT} and ${PREVIEW_BACKEND_PORT} over SSH."
exec npm exec --yes --package="fern-api@${FERN_VERSION}" -- fern docs dev \
  --port "${PREVIEW_PORT}" --backend-port "${PREVIEW_BACKEND_PORT}"
