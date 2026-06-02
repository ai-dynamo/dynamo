#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"

PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
VLLM_OMNI_VERSION="${VLLM_OMNI_REF#v}"
VLLM_OMNI_PATCH_DIR="${VLLM_OMNI_PATCH_DIR:-/tmp/vllm_patches/vllm-omni-v${VLLM_OMNI_VERSION}}"

cleanup() {
  rm -rf "${PROTECTED_CONSTRAINTS}"
}

trap cleanup EXIT

python3 - "${VLLM_OMNI_PROTECTED_PACKAGES_FILE}" <<'PY' > "${PROTECTED_CONSTRAINTS}"
import importlib.metadata as md
from pathlib import Path
import sys

for raw_line in Path(sys.argv[1]).read_text().splitlines():
    name = raw_line.strip()
    if not name or name.startswith("#"):
        continue
    try:
        dist = md.distribution(name)
    except Exception:
        continue
    project_name = dist.metadata.get("Name") or name
    print(f"{project_name}=={dist.version}")
PY

export VLLM_OMNI_TARGET_DEVICE

# Use --system flag only for CUDA (system Python), omit for CPU/XPU (venv)
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  uv pip install --system \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "vllm-omni==${VLLM_OMNI_VERSION}"
else
  uv pip install \
    --prerelease=allow \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    "vllm-omni==${VLLM_OMNI_VERSION}"
fi

if [ -d "${VLLM_OMNI_PATCH_DIR}" ]; then
  SITE_PACKAGES="$(python3 - <<'PY'
import site

paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
  for patch_file in "${VLLM_OMNI_PATCH_DIR}"/*.patch; do
    [ -e "${patch_file}" ] || continue
    echo "Applying vLLM-Omni patch: ${patch_file}"
    if command -v git >/dev/null 2>&1; then
      (cd "${SITE_PACKAGES}" && git apply --ignore-whitespace -p0 "${patch_file}")
    else
      (cd "${SITE_PACKAGES}" && patch -p0 -l -i "${patch_file}")
    fi
  done
fi
