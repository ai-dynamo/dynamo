#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${VLLM_OMNI_REF:?VLLM_OMNI_REF must be set}"

VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
VLLM_OMNI_PROTECTED_PACKAGES_FILE="${VLLM_OMNI_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"
VLLM_OMNI_TARGET_DEVICE="${VLLM_OMNI_TARGET_DEVICE:-cuda}"

REPO_DIR="$(mktemp -d /tmp/vllm-omni.XXXXXX)"
PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-openai-protected.XXXXXX.txt)"
VLLM_CLI_BACKUP="$(mktemp /tmp/vllm-openai-cli.XXXXXX)"

cleanup() {
  rm -rf "${REPO_DIR}" "${PROTECTED_CONSTRAINTS}" "${VLLM_CLI_BACKUP}"
}

trap cleanup EXIT

cp /usr/local/bin/vllm "${VLLM_CLI_BACKUP}"
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
uv pip install --system --no-deps "${REPO_DIR}"
uv pip install --system \
  --constraints "${PROTECTED_CONSTRAINTS}" \
  --requirement "${REPO_DIR}/requirements/common.txt" \
  "onnxruntime>=1.23.2"
install -m 755 "${VLLM_CLI_BACKUP}" /usr/local/bin/vllm
