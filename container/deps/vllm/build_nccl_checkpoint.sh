#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ -z "${NCCL_CHECKPOINT_VERSION:-}" ]; then
  echo "NCCL_CHECKPOINT_VERSION is empty; skipping NCCLCheckpoint shim build."
  exit 0
fi

PREFIX="${NCCL_CHECKPOINT_PREFIX:-/opt/nccl-checkpoint}"
SRC_DIR="${NCCL_CHECKPOINT_SRC_DIR:-/tmp/nccl-src}"
ARTIFACT_ROOT="${NCCL_CHECKPOINT_ARTIFACT_ROOT:-/tmp/nccl-source-artifact}"
ARTIFACT_SRC="${ARTIFACT_ROOT}/opt/nccl-source"
ARTIFACT_PROVENANCE="${ARTIFACT_ROOT}/opt/nccl-source-provenance"
SOURCE_BUNDLE="${ARTIFACT_PROVENANCE}/source.bundle"
SOURCE_PROVENANCE="${ARTIFACT_PROVENANCE}/provenance.json"
SHIM="${PREFIX}/lib/libnccl-checkpoint-shim.so"
PYNCCL_SMOKE_CHECK="${PYNCCL_SMOKE_CHECK:-/usr/local/lib/validate_pynccl_checkpoint_binding.py}"
RUNTIME_PROVENANCE="${NCCL_CHECKPOINT_RUNTIME_PROVENANCE:-/opt/dynamo/source-provenance.txt}"

EXPECTED_SOURCE_IMAGE="dynamoci.azurecr.io/ai-dynamo/nccl-source@sha256:5502f117103a84d8738f822f98c1d591d9b75fc72b14031d9e57af3a8db5b10c"
EXPECTED_SOURCE_BRANCH="schwinns/nccl-2.29.7-checkpoint-shim"
EXPECTED_SOURCE_SHA="0c112d6e5a81d5b5c39e3b295aa06282a5538979"
EXPECTED_SOURCE_TREE="64912568e8b1ea07e7f0dd2c80cf9c4db986eace"
EXPECTED_BASE_TAG="v2.29.7-1"
EXPECTED_BASE_SHA="b81d6a5a3d2fa95ad11f6453c51cd6a6ba19f9b8"
EXPECTED_CHECKPOINT_TREE="f43d560f98e687b0f175350b6ea51f054cbcb654"

if [ "${NCCL_CHECKPOINT_SOURCE_IMAGE:-}" != "${EXPECTED_SOURCE_IMAGE}" ]; then
  echo "NCCL checkpoint build requires ${EXPECTED_SOURCE_IMAGE}." >&2
  exit 1
fi
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

test -d "${ARTIFACT_SRC}"
test -f "${SOURCE_BUNDLE}"
test -f "${SOURCE_PROVENANCE}"

EXPECTED_SOURCE_BRANCH="${EXPECTED_SOURCE_BRANCH}" \
EXPECTED_SOURCE_SHA="${EXPECTED_SOURCE_SHA}" \
EXPECTED_SOURCE_TREE="${EXPECTED_SOURCE_TREE}" \
EXPECTED_BASE_TAG="${EXPECTED_BASE_TAG}" \
EXPECTED_BASE_SHA="${EXPECTED_BASE_SHA}" \
EXPECTED_CHECKPOINT_TREE="${EXPECTED_CHECKPOINT_TREE}" \
  python3 - "${SOURCE_PROVENANCE}" <<'PY'
import json
import os
import sys

expected = {
    "schema_version": 1,
    "artifact_kind": "nccl-source",
    "source_repository_visibility": "private-internal",
    "source_branch": os.environ["EXPECTED_SOURCE_BRANCH"],
    "source_commit": os.environ["EXPECTED_SOURCE_SHA"],
    "source_tree": os.environ["EXPECTED_SOURCE_TREE"],
    "base_tag": os.environ["EXPECTED_BASE_TAG"],
    "base_commit": os.environ["EXPECTED_BASE_SHA"],
    "checkpoint_subtree": "contrib/nccl_checkpoint",
    "checkpoint_tree": os.environ["EXPECTED_CHECKPOINT_TREE"],
}
with open(sys.argv[1]) as provenance_file:
    actual = json.load(provenance_file)
if actual != expected:
    raise SystemExit(
        f"NCCL source provenance does not match:\n"
        f"expected={expected!r}\nactual={actual!r}"
    )
print(f"Verified NCCL source provenance JSON: {actual!r}")
PY

verify_repo=/tmp/nccl-source-verify.git
rm -rf "${verify_repo}"
git clone --quiet --bare "${SOURCE_BUNDLE}" "${verify_repo}"
bundle_ref="refs/heads/${EXPECTED_SOURCE_BRANCH}"
bundle_head="$(git -C "${verify_repo}" rev-parse "${bundle_ref}^{commit}")"
bundle_parent="$(git -C "${verify_repo}" rev-parse "${bundle_head}^")"
bundle_tree="$(git -C "${verify_repo}" rev-parse "${bundle_head}^{tree}")"
bundle_base_tag="$(
  git -C "${verify_repo}" rev-parse \
    "refs/tags/${EXPECTED_BASE_TAG}^{commit}"
)"
bundle_checkpoint_tree="$(
  git -C "${verify_repo}" rev-parse \
    "${bundle_head}:contrib/nccl_checkpoint"
)"
test "${bundle_head}" = "${EXPECTED_SOURCE_SHA}"
test "${bundle_parent}" = "${EXPECTED_BASE_SHA}"
test "${bundle_base_tag}" = "${EXPECTED_BASE_SHA}"
test "${bundle_tree}" = "${EXPECTED_SOURCE_TREE}"
test "${bundle_checkpoint_tree}" = "${EXPECTED_CHECKPOINT_TREE}"
test "$(
  git -C "${verify_repo}" rev-list --count \
    "${EXPECTED_BASE_SHA}..${EXPECTED_SOURCE_SHA}"
)" = 1
test -z "$(
  git -C "${verify_repo}" diff --name-only \
    "${EXPECTED_BASE_SHA}..${EXPECTED_SOURCE_SHA}" -- \
    . ':(exclude)contrib/nccl_checkpoint'
)"

rm -rf "${SRC_DIR}"
cp -a "${ARTIFACT_SRC}" "${SRC_DIR}"
archive_repo=/tmp/nccl-source-archive.git
rm -rf "${archive_repo}"
git init --quiet "${archive_repo}"
cp -a "${SRC_DIR}/." "${archive_repo}/"
git -C "${archive_repo}" add --all --force
archive_tree="$(git -C "${archive_repo}" write-tree)"
archive_checkpoint_tree="$(
  git -C "${archive_repo}" rev-parse \
    "${archive_tree}:contrib/nccl_checkpoint"
)"
test "${archive_tree}" = "${EXPECTED_SOURCE_TREE}"
test "${archive_checkpoint_tree}" = "${EXPECTED_CHECKPOINT_TREE}"
rm -rf "${verify_repo}" "${archive_repo}"
echo "Verified NCCL source commit ${bundle_head} with parent ${bundle_parent}"
echo "Verified NCCL base tag ${EXPECTED_BASE_TAG} at ${bundle_base_tag}"
echo "Verified archived NCCL source tree ${archive_tree}"
echo "Verified checkpoint subtree ${archive_checkpoint_tree}"

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
  echo "nccl_source_image=${NCCL_CHECKPOINT_SOURCE_IMAGE}"
  echo "nccl_source_branch=${EXPECTED_SOURCE_BRANCH}"
  echo "nccl_source_sha=${EXPECTED_SOURCE_SHA}"
  echo "nccl_source_tree=${EXPECTED_SOURCE_TREE}"
  echo "nccl_base_tag=${EXPECTED_BASE_TAG}"
  echo "nccl_base_sha=${EXPECTED_BASE_SHA}"
  echo "nccl_checkpoint_tree=${EXPECTED_CHECKPOINT_TREE}"
} >> "${RUNTIME_PROVENANCE}"
echo "Installed NCCLCheckpoint shim at ${SHIM}"
