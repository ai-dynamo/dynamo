#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PREFIX="${NCCL_CHECKPOINT_PREFIX:-/opt/nccl-checkpoint}"
SHIM="${PREFIX}/lib/libnccl-checkpoint-shim.so"
LD_PRELOAD_FILE="${NCCL_CHECKPOINT_LD_PRELOAD_FILE:-/etc/ld.so.preload}"

if [ -z "${NCCL_CHECKPOINT_VERSION:-}" ]; then
  if [ -f "${LD_PRELOAD_FILE}" ]; then
    preload_tmp="$(mktemp)"
    grep -Fvx -- "${SHIM}" "${LD_PRELOAD_FILE}" > "${preload_tmp}" || true
    if [ -s "${preload_tmp}" ]; then
      cat "${preload_tmp}" > "${LD_PRELOAD_FILE}"
    else
      rm -f "${LD_PRELOAD_FILE}"
    fi
    rm -f "${preload_tmp}"
  fi
  rm -rf "${PREFIX}"
  uv pip uninstall --system nccl_checkpoint
  python3 -c 'import importlib.util; assert importlib.util.find_spec("nccl_checkpoint") is None'
  echo "NCCL_CHECKPOINT_VERSION is empty; removed inherited NCCLCheckpoint shim."
  exit 0
fi

SRC_DIR="${NCCL_CHECKPOINT_SRC_DIR:-/tmp/nccl-src}"
PYNCCL_SMOKE_CHECK="${PYNCCL_SMOKE_CHECK:-/usr/local/lib/validate_pynccl_checkpoint_binding.py}"
RUNTIME_PROVENANCE="${NCCL_CHECKPOINT_RUNTIME_PROVENANCE:-/opt/dynamo/source-provenance.txt}"

NCCL_PUBLIC_GIT_URL="https://github.com/NVIDIA/nccl.git"
NCCL_CORE_SHA="b81d6a5a3d2fa95ad11f6453c51cd6a6ba19f9b8"
NCCL_CORE_TREE="d61e2c86cfc507969000a963782b2e2231ff1475"
NCCL_SHIM_SOURCE_SHA="a2e67d265b3c52172f20452a909f7eaa6a0f1328"
NCCL_CHECKPOINT_TREE="f43d560f98e687b0f175350b6ea51f054cbcb654"
NCCL_COMPOSED_TREE="64912568e8b1ea07e7f0dd2c80cf9c4db986eace"

if [ -z "${CUDA_MAJOR:-}" ]; then
  echo "CUDA_MAJOR is required when NCCL_CHECKPOINT_VERSION is set." >&2
  exit 1
fi
uv pip install --system --force-reinstall --no-deps \
  "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_CHECKPOINT_VERSION}"

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

rm -rf "${SRC_DIR}"
git init --quiet "${SRC_DIR}"
git -C "${SRC_DIR}" remote add origin "${NCCL_PUBLIC_GIT_URL}"

git -C "${SRC_DIR}" fetch --no-tags --depth=1 origin "${NCCL_CORE_SHA}"
resolved_core_sha="$(git -C "${SRC_DIR}" rev-parse "FETCH_HEAD^{commit}")"
test "${resolved_core_sha}" = "${NCCL_CORE_SHA}"
git -C "${SRC_DIR}" checkout --quiet --detach "${resolved_core_sha}"
resolved_core_tree="$(git -C "${SRC_DIR}" rev-parse "HEAD^{tree}")"
test "${resolved_core_tree}" = "${NCCL_CORE_TREE}"
if git -C "${SRC_DIR}" cat-file -e \
    "${NCCL_CORE_SHA}:contrib/nccl_checkpoint" 2>/dev/null; then
  echo "Pinned NCCL core unexpectedly contains contrib/nccl_checkpoint." >&2
  exit 1
fi

git -C "${SRC_DIR}" fetch --no-tags --depth=1 origin \
  "${NCCL_SHIM_SOURCE_SHA}"
resolved_shim_sha="$(git -C "${SRC_DIR}" rev-parse "FETCH_HEAD^{commit}")"
test "${resolved_shim_sha}" = "${NCCL_SHIM_SOURCE_SHA}"
resolved_checkpoint_tree="$(
  git -C "${SRC_DIR}" rev-parse \
    "${resolved_shim_sha}:contrib/nccl_checkpoint"
)"
test "${resolved_checkpoint_tree}" = "${NCCL_CHECKPOINT_TREE}"

git -C "${SRC_DIR}" archive "${resolved_shim_sha}" \
  contrib/nccl_checkpoint | tar -x -C "${SRC_DIR}"
git -C "${SRC_DIR}" add --force contrib/nccl_checkpoint
composed_tree="$(git -C "${SRC_DIR}" write-tree)"
composed_checkpoint_tree="$(
  git -C "${SRC_DIR}" rev-parse \
    "${composed_tree}:contrib/nccl_checkpoint"
)"
test "${composed_tree}" = "${NCCL_COMPOSED_TREE}"
test "${composed_checkpoint_tree}" = "${NCCL_CHECKPOINT_TREE}"
echo "Verified public NCCL core ${resolved_core_sha} tree ${resolved_core_tree}"
echo "Verified public shim source ${resolved_shim_sha}"
echo "Verified checkpoint subtree ${composed_checkpoint_tree}"
echo "Verified composed NCCL source tree ${composed_tree}"

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
{
  echo "nccl_public_git_url=${NCCL_PUBLIC_GIT_URL}"
  echo "nccl_core_sha=${NCCL_CORE_SHA}"
  echo "nccl_core_tree=${NCCL_CORE_TREE}"
  echo "nccl_shim_source_sha=${NCCL_SHIM_SOURCE_SHA}"
  echo "nccl_checkpoint_tree=${NCCL_CHECKPOINT_TREE}"
  echo "nccl_composed_tree=${NCCL_COMPOSED_TREE}"
} >> "${RUNTIME_PROVENANCE}"
echo "Installed NCCLCheckpoint shim at ${SHIM}"
