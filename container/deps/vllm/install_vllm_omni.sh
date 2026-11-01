#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"
VLLM_OMNI_PROTECTED_OVERRIDES_FILE="${VLLM_OMNI_PROTECTED_OVERRIDES_FILE:-/tmp/vllm_omni_protected_overrides.txt}"

PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
PROTECTED_OVERRIDES="$(mktemp /tmp/vllm-openai-overrides.XXXXXX.txt)"
VLLM_OMNI_VERSION="${VLLM_OMNI_REF#v}"

cleanup() {
  rm -rf "${PROTECTED_CONSTRAINTS}" "${PROTECTED_OVERRIDES}"
}

trap cleanup EXIT

# Print name==installed-version for each installed package named in a list file.
freeze_installed() {
python3 - "$1" <<'PY'
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
}

freeze_installed "${VLLM_OMNI_PROTECTED_PACKAGES_FILE}" > "${PROTECTED_CONSTRAINTS}"

# XPU only: vllm-openai-xpu force-upgrades torch's ==-pinned oneAPI runtime as its
# last build step, so a solve with torch frozen would downgrade it again. Hold the
# installed versions as overrides; see protected_overrides.txt.
# TODO: remove once the base image declares these packages through UV_OVERRIDE.
OVERRIDE_ARGS=()
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "xpu" ]; then
  freeze_installed "${VLLM_OMNI_PROTECTED_OVERRIDES_FILE}" > "${PROTECTED_OVERRIDES}"
  OVERRIDE_ARGS=(--overrides "${PROTECTED_OVERRIDES}")
fi

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
    "${OVERRIDE_ARGS[@]}" \
    "vllm-omni==${VLLM_OMNI_VERSION}"
fi
