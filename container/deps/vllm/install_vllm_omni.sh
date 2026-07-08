#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"

PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
VLLM_OMNI_VERSION="${VLLM_OMNI_REF#v}"

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

# Cherry-pick vllm-project/vllm-omni#4568 onto the released wheel.
#
# vLLM-Omni globally monkeypatches vllm.v1.request.Request with its OmniRequest
# subclass at import time, and the test suite imports vllm_omni for collection,
# so this applies to every vLLM worker in the image -- not just omni modes. In
# the released v0.23.0rc1, OmniRequest.__init__ still declares `*args` after its
# named parameters, so vLLM 0.23's positional Request(...) construction misbinds
# the arguments and EngineCore initialization fails for all vLLM workers. The fix
# moves `*args` to the front and forwards cleanly. Drop this once a vllm-omni
# release includes the change.
# https://github.com/vllm-project/vllm-omni/commit/17cf60a63d240608653c4532084a4c00d6f02216
VLLM_OMNI_CHERRY_PICK_COMMIT="17cf60a63d240608653c4532084a4c00d6f02216"

omni_site="$(python3 -c 'import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec("vllm_omni").origin)))')"
full_patch="$(mktemp /tmp/vllm-omni-commit.XXXXXX.patch)"
cherry_pick_patch="$(mktemp /tmp/vllm-omni-cherry-pick.XXXXXX.patch)"

curl -fsSL \
  "https://github.com/vllm-project/vllm-omni/commit/${VLLM_OMNI_CHERRY_PICK_COMMIT}.patch" \
  -o "${full_patch}"
# Keep only the vllm_omni/request.py hunk; the commit's new test file is not part
# of the installed wheel.
awk '/^diff --git a\/vllm_omni\/request.py/{f=1} f' "${full_patch}" > "${cherry_pick_patch}"

if [ ! -s "${cherry_pick_patch}" ]; then
  echo "ERROR: could not extract request.py hunk from vllm-omni commit ${VLLM_OMNI_CHERRY_PICK_COMMIT}" >&2
  exit 1
fi

if patch -p1 -d "${omni_site}" --forward --dry-run < "${cherry_pick_patch}" >/dev/null 2>&1; then
  patch -p1 -d "${omni_site}" --forward < "${cherry_pick_patch}"
  echo "Applied vllm-omni cherry-pick ${VLLM_OMNI_CHERRY_PICK_COMMIT}"
elif patch -p1 -d "${omni_site}" --reverse --dry-run < "${cherry_pick_patch}" >/dev/null 2>&1; then
  echo "vllm-omni cherry-pick ${VLLM_OMNI_CHERRY_PICK_COMMIT} already present; skipping"
else
  echo "ERROR: vllm-omni cherry-pick ${VLLM_OMNI_CHERRY_PICK_COMMIT} does not apply cleanly to the installed package" >&2
  exit 1
fi

rm -f "${full_patch}" "${cherry_pick_patch}"
