#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${CUDA_VERSION:?CUDA_VERSION must be set}"
: "${LMCACHE_REF:?LMCACHE_REF must be set}"
: "${TORCH_LIB_DIR:?TORCH_LIB_DIR must be set}"

CUDA_MAJOR="${CUDA_VERSION%%.*}"
if [[ "${CUDA_MAJOR}" != "13" ]]; then
  exit 0
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Skipping LMCache source rebuild on non-x86_64 CUDA 13 image"
  exit 0
fi

CUDA_VERSION_DASH="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr '.' '-')"
TORCH_CUDA_ARCH_LIST="${LMCACHE_TORCH_CUDA_ARCH_LIST:-8.0 8.9 9.0 10.0 12.0}"
BUILD_PACKAGES=(
  build-essential
  "libcusparse-dev-${CUDA_VERSION_DASH}"
  "libcublas-dev-${CUDA_VERSION_DASH}"
  "libcusolver-dev-${CUDA_VERSION_DASH}"
)

WORKDIR="$(mktemp -d /tmp/lmcache.XXXXXX)"
cleanup() {
  rm -rf "${WORKDIR}"
  apt-get purge -y --auto-remove "${BUILD_PACKAGES[@]}" >/dev/null 2>&1 || true
  rm -rf /var/lib/apt/lists/*
}
trap cleanup EXIT

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${BUILD_PACKAGES[@]}"

git clone --depth 1 --branch "v${LMCACHE_REF}" https://github.com/LMCache/LMCache.git "${WORKDIR}/src"
cd "${WORKDIR}/src"

uv pip install --system --requirement requirements/build.txt
export TORCH_CUDA_ARCH_LIST
CUDA_HOME=/usr/local/cuda \
LDFLAGS="-Wl,-rpath,${TORCH_LIB_DIR}" \
uv pip install --system --no-deps --force-reinstall --no-build-isolation .

python3 - <<'PY'
import pathlib
import subprocess

site = pathlib.Path("/usr/local/lib")
matches = sorted(site.glob("python*/dist-packages/lmcache/c_ops*.so"))
assert matches, "lmcache c_ops shared object not found after source rebuild"
ldd = subprocess.check_output(["ldd", str(matches[-1])], text=True)
assert "libcudart.so.13" in ldd, ldd
PY
