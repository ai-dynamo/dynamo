#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install vLLM-Omni either from a PyPI release or from a git ref.
#
#   VLLM_OMNI_GIT_REF set  -> install from git (commit / branch / tag) on
#                             VLLM_OMNI_GIT_URL. Use this when PyPI has no
#                             release matching the pinned vLLM (e.g. vLLM 0.22
#                             before a vllm-omni 0.22 release is published).
#                             Requires `git` on PATH.
#   otherwise              -> install vllm-omni==${VLLM_OMNI_REF#v} from PyPI.
#
# Both modes resolve vllm-omni's dependencies, but hold the packages already
# solved in the upstream vLLM image to their installed versions via the
# protected-packages list:
#   * PyPI release  -> as --constraints (the release's bounds already agree
#                      with the image, so a constraint is enough).
#   * git ref       -> as --override (a ref ahead of any release often declares
#                      bounds that conflict with the image's pins, e.g.
#                      transformers; --override forces the image's versions
#                      while still pulling vllm-omni's other deps such as aenum).
#
# VLLM_OMNI_GIT_REF takes precedence over VLLM_OMNI_REF when both are set.
VLLM_OMNI_GIT_REF="${VLLM_OMNI_GIT_REF:-}"
VLLM_OMNI_GIT_URL="${VLLM_OMNI_GIT_URL:-https://github.com/vllm-project/vllm-omni}"
VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"

PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
cleanup() { rm -rf "${PROTECTED_CONSTRAINTS}"; }
trap cleanup EXIT

# Pin every still-installed protected package to its current version.
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

INSTALL_FLAGS=(--prerelease=allow)

# --system flag only for CUDA (system Python); omit for CPU/XPU (venv).
export VLLM_OMNI_TARGET_DEVICE
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  INSTALL_FLAGS+=(--system)
fi

if [ -n "${VLLM_OMNI_GIT_REF}" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "VLLM_OMNI_GIT_REF=${VLLM_OMNI_GIT_REF} requires git, which is not installed" >&2
    exit 1
  fi
  VLLM_OMNI_SPEC="vllm-omni @ git+${VLLM_OMNI_GIT_URL}@${VLLM_OMNI_GIT_REF}"
  INSTALL_FLAGS+=(--override "${PROTECTED_CONSTRAINTS}")
  echo "Installing vLLM-Omni from git: ${VLLM_OMNI_GIT_URL}@${VLLM_OMNI_GIT_REF}"
else
  : "${VLLM_OMNI_REF:?VLLM_OMNI_REF or VLLM_OMNI_GIT_REF must be set}"
  VLLM_OMNI_SPEC="vllm-omni==${VLLM_OMNI_REF#v}"
  INSTALL_FLAGS+=(--constraints "${PROTECTED_CONSTRAINTS}")
  echo "Installing vLLM-Omni from PyPI: ${VLLM_OMNI_SPEC}"
fi

uv pip install "${INSTALL_FLAGS[@]}" "${VLLM_OMNI_SPEC}"
