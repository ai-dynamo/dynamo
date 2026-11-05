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

# vLLM-Omni 0.33 registers its experimental Physis pipelines (vllm-omni#9154) eagerly,
# so every Omni process imports PyAV, which the CUDA image removes under the codec
# policy, and Qwen3-TTS fails requests when stage 0 uses FlashInfer. Remove those
# registrations; remove this once vllm-project/vllm-omni#9939 is fixed.
if [ "${VLLM_OMNI_TARGET_DEVICE}" = "cuda" ]; then
  python3 - <<'PY'
import importlib.util
import py_compile
from pathlib import Path

pkg = Path(importlib.util.find_spec("vllm_omni").submodule_search_locations[0])
path = pkg / "config" / "pipeline_registry.py"
lines = path.read_text().splitlines(keepends=True)
expected = [
    "from vllm_omni.diffusion.models.physis.h3.pipeline_config import PHYSIS_H3_PIPELINE",
    "from vllm_omni.diffusion.models.physis.ltx.pipeline_config import PHYSIS_LTX_PIPELINE",
    "from vllm_omni.diffusion.models.physis.wan.pipeline_config import PHYSIS_WAN_PIPELINE",
    '"physis_h3": PHYSIS_H3_PIPELINE,',
    '"physis_ltx": PHYSIS_LTX_PIPELINE,',
    '"physis_wan": PHYSIS_WAN_PIPELINE,',
]
physis = [line.strip() for line in lines if "physis" in line.lower()]
if physis != expected:
    raise SystemExit(f"{path}: Physis lines {physis} differ from {expected}; update or remove this patch")
kept = "".join(line for line in lines if "physis" not in line.lower())
path.write_text("# Modified by NVIDIA Dynamo: experimental Physis pipelines removed.\n" + kept)
py_compile.compile(str(path), doraise=True)
PY
  # The image purges PyAV in a later step; check Omni's pipeline registry loads without it.
  python3 -c "import sys; sys.modules['av'] = None; import vllm_omni.config.pipeline_registry"
fi
