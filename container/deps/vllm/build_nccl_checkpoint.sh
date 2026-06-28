#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ -z "${NCCL_CHECKPOINT_GIT_URL:-}" ]; then
  echo "NCCL_CHECKPOINT_GIT_URL is empty; skipping NCCLCheckpoint shim build."
  exit 0
fi

PREFIX="${NCCL_CHECKPOINT_PREFIX:-/opt/nccl-checkpoint}"
SRC_DIR="${NCCL_CHECKPOINT_SRC_DIR:-/tmp/nccl-src}"
SHIM="${PREFIX}/lib/libnccl-checkpoint-shim.so"
PYNCCL_SMOKE_CHECK="${PYNCCL_SMOKE_CHECK:-/usr/local/lib/validate_pynccl_checkpoint_binding.py}"

if [ -n "${NCCL_CHECKPOINT_VERSION:-}" ]; then
  if [ -z "${CUDA_MAJOR:-}" ]; then
    echo "CUDA_MAJOR is required when NCCL_CHECKPOINT_VERSION is set." >&2
    exit 1
  fi
  uv pip install --system --force-reinstall --no-deps \
    "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_CHECKPOINT_VERSION}"
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

parse_header_define() {
  local header="$1"
  local name="$2"
  sed -nE "s/^[[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+([0-9]+).*/\\1/p" \
    "${header}" | head -n1
}

parse_make_var() {
  local file="$1"
  local name="$2"
  local value
  value="$(
    sed -nE "s/^[[:space:]]*${name}[[:space:]]*:?=[[:space:]]*([^#]+).*/\\1/p" \
      "${file}" | head -n1
  )"
  trim "${value}"
}

version_code() {
  local major="$1"
  local minor="$2"
  local patch="$3"
  if [ "${major}" -le 2 ] && [ "${minor}" -le 8 ]; then
    printf '%d\n' "$((major * 1000 + minor * 100 + patch))"
  else
    printf '%d\n' "$((major * 10000 + minor * 100 + patch))"
  fi
}

find_installed_nccl_header() {
  local header
  for header in \
    "/usr/local/lib/python${PYTHON_VERSION:-3.12}/dist-packages/nvidia/nccl/include/nccl.h" \
    "/usr/local/lib/python${PYTHON_VERSION:-3.12}/site-packages/nvidia/nccl/include/nccl.h" \
    /usr/local/cuda/include/nccl.h \
    /usr/include/nccl.h; do
    if [ -f "${header}" ]; then
      printf '%s\n' "${header}"
      return 0
    fi
  done

  find /usr/local/lib -type f \
    \( -path '*/dist-packages/nvidia/nccl/include/nccl.h' \
       -o -path '*/site-packages/nvidia/nccl/include/nccl.h' \) \
    -print -quit
}

installed_header="$(find_installed_nccl_header)"
if [ -z "${installed_header}" ] || [ ! -f "${installed_header}" ]; then
  echo "Unable to find installed nccl.h for NCCLCheckpoint version check." >&2
  exit 1
fi

installed_major="$(parse_header_define "${installed_header}" NCCL_MAJOR)"
installed_minor="$(parse_header_define "${installed_header}" NCCL_MINOR)"
installed_patch="$(parse_header_define "${installed_header}" NCCL_PATCH)"
installed_code="$(parse_header_define "${installed_header}" NCCL_VERSION_CODE)"
if [ -z "${installed_major}" ] || [ -z "${installed_minor}" ] || [ -z "${installed_patch}" ]; then
  echo "Unable to parse NCCL version from ${installed_header}." >&2
  exit 1
fi
if [ -z "${installed_code}" ]; then
  installed_code="$(version_code "${installed_major}" "${installed_minor}" "${installed_patch}")"
fi

cuda_packages=()
if [ -n "${CUDA_VERSION:-}" ]; then
  cuda_minor_dash="${CUDA_VERSION%%.*}-${CUDA_VERSION#*.}"
  cuda_minor_dash="${cuda_minor_dash%.*}"
  cuda_packages+=("cuda-cudart-dev-${cuda_minor_dash}")
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  git \
  build-essential \
  pkg-config \
  libhiredis-dev \
  "${cuda_packages[@]}"
rm -rf /var/lib/apt/lists/*

auto_ref_for_installed_version() {
  local url="$1"
  local candidate
  for candidate in \
    "v${installed_major}.${installed_minor}.${installed_patch}-1" \
    "v${installed_major}.${installed_minor}.${installed_patch}" \
    "v${installed_major}.${installed_minor}.${installed_patch}-0"; do
    if git ls-remote --exit-code --tags "${url}" "refs/tags/${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if git ls-remote --exit-code --heads "${url}" "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
}

ref="${NCCL_CHECKPOINT_GIT_REF:-}"
sha="${NCCL_CHECKPOINT_GIT_SHA:-}"
if [ -z "${ref}" ] && [ -z "${sha}" ]; then
  ref="$(auto_ref_for_installed_version "${NCCL_CHECKPOINT_GIT_URL}")"
  if [ -z "${ref}" ]; then
    echo "NCCL_CHECKPOINT_GIT_REF is empty and no tag matching installed NCCL ${installed_major}.${installed_minor}.${installed_patch} was found." >&2
    exit 1
  fi
  echo "Auto-selected NCCL source ref ${ref} for installed NCCL ${installed_major}.${installed_minor}.${installed_patch}."
fi

rm -rf "${SRC_DIR}"
if [ -n "${ref}" ]; then
  git clone --recurse-submodules --branch "${ref}" --depth 1 \
    "${NCCL_CHECKPOINT_GIT_URL}" "${SRC_DIR}"
else
  git clone --recurse-submodules "${NCCL_CHECKPOINT_GIT_URL}" "${SRC_DIR}"
fi

cd "${SRC_DIR}"
if [ -n "${sha}" ]; then
  git fetch --depth 1 origin "${sha}" || true
  resolved_sha="$(git rev-parse "${sha}^{commit}")"
  git checkout --detach "${resolved_sha}"
  test "$(git rev-parse HEAD)" = "${resolved_sha}"
  git submodule update --init --recursive
fi

version_file="${SRC_DIR}/makefiles/version.mk"
if [ ! -f "${version_file}" ]; then
  echo "NCCL source tree is missing ${version_file}." >&2
  exit 1
fi
source_major="$(parse_make_var "${version_file}" NCCL_MAJOR)"
source_minor="$(parse_make_var "${version_file}" NCCL_MINOR)"
source_patch="$(parse_make_var "${version_file}" NCCL_PATCH)"
if [ -z "${source_major}" ] || [ -z "${source_minor}" ] || [ -z "${source_patch}" ]; then
  echo "Unable to parse NCCL source version from ${version_file}." >&2
  exit 1
fi
source_code="$(version_code "${source_major}" "${source_minor}" "${source_patch}")"

if [ "${source_code}" != "${installed_code}" ]; then
  cat >&2 <<EOF
NCCLCheckpoint source version does not match installed NCCL.
  installed: ${installed_major}.${installed_minor}.${installed_patch} (${installed_code}) from ${installed_header}
  source:    ${source_major}.${source_minor}.${source_patch} (${source_code}) from ${SRC_DIR}
EOF
  exit 1
fi

cd "${SRC_DIR}/contrib/nccl_checkpoint"
make -j"$(nproc)" NCCL_SRC="${SRC_DIR}/src" PREFIX="${PREFIX}" \
  CUDA_VERSION="${CUDA_VERSION:-}"
make install NCCL_SRC="${SRC_DIR}/src" PREFIX="${PREFIX}" \
  CUDA_VERSION="${CUDA_VERSION:-}"
uv pip install --system --no-deps ./python
test -f "${SHIM}"
test -f "${PYNCCL_SMOKE_CHECK}"
NCCL_CHECKPOINT_SHIM="${SHIM}" \
EXPECTED_NCCL_VERSION="${NCCL_CHECKPOINT_VERSION:-}" \
EXPECTED_TORCH_VERSION="${VLLM_TORCH_VERSION:-}" \
EXPECTED_TORCH_LOCAL_VERSION="${VLLM_TORCH_BACKEND:-${VLLM_PRECOMPILED_WHEEL_VARIANT:-}}" \
LD_PRELOAD="${SHIM}${LD_PRELOAD:+:${LD_PRELOAD}}" \
  python3 "${PYNCCL_SMOKE_CHECK}"

rm -rf "${SRC_DIR}"
echo "Installed NCCLCheckpoint shim at ${SHIM}"
