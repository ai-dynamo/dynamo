#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${LMCACHE_REF:?LMCACHE_REF must be set}"

src_dir="$(mktemp -d /tmp/lmcache-xpu-src.XXXXXX)"

cleanup() {
  rm -rf "${src_dir}"
}

trap cleanup EXIT

git clone https://github.com/LMCache/LMCache.git "${src_dir}"
git -C "${src_dir}" checkout "${LMCACHE_REF}"

uv pip install --requirement "${src_dir}/requirements/build.txt"
BUILD_WITH_SYCL=1 uv pip install --no-build-isolation "${src_dir}"

python3 - <<'PY'
import importlib.metadata as md

print(f"Installed LMCache {md.version('lmcache')}")
PY
