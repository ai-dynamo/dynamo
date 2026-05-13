#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"
VLLM_OMNI_TARGET_DEVICE="${VLLM_OMNI_TARGET_DEVICE:-cuda}"

# Find vllm installation location using pip list
VLLM_INSTALL_DIR=$(pip list -v | grep '^vllm ' | awk '{print $NF}')
if [ -z "${VLLM_INSTALL_DIR}" ]; then
  echo "ERROR: Could not find vllm installation via 'pip list -v'" >&2
  exit 1
fi

# Determine vllm CLI path based on device and environment
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  # CUDA uses system path
  VLLM_CLI_PATH="/usr/local/bin/vllm"
elif [ -n "${VIRTUAL_ENV}" ]; then
  # CPU/XPU use venv's bin directory
  VLLM_CLI_PATH="${VIRTUAL_ENV}/bin/vllm"
else
  # Fallback: derive from install dir
  VLLM_CLI_PATH="$(dirname "${VLLM_INSTALL_DIR}")/../../bin/vllm"
fi

if [ ! -f "${VLLM_CLI_PATH}" ]; then
  echo "ERROR: vllm CLI not found at ${VLLM_CLI_PATH}" >&2
  exit 1
fi

REPO_DIR="$(mktemp -d /tmp/vllm-omni.XXXXXX)"
PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
VLLM_CLI_BACKUP="$(mktemp /tmp/vllm-openai-cli.XXXXXX)"

cleanup() {
  rm -rf "${REPO_DIR}" "${PROTECTED_CONSTRAINTS}" "${VLLM_CLI_BACKUP}"
}

trap cleanup EXIT

cp "${VLLM_CLI_PATH}" "${VLLM_CLI_BACKUP}"
git clone --depth 1 --branch "${VLLM_OMNI_REF}" "${VLLM_OMNI_REPO}" "${REPO_DIR}"

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
  uv pip install --system --no-deps "${REPO_DIR}"
  uv pip install --system \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    --requirement "${REPO_DIR}/requirements/common.txt" \
    "onnxruntime>=1.23.2"
else
  uv pip install --no-deps "${REPO_DIR}"
  uv pip install \
    --constraints "${PROTECTED_CONSTRAINTS}" \
    --requirement "${REPO_DIR}/requirements/common.txt" \
    "onnxruntime>=1.23.2"
fi

install -m 755 "${VLLM_CLI_BACKUP}" "${VLLM_CLI_PATH}"
