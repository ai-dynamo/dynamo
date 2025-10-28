#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the KVBM wheel.

# This script builds KVBM pip wheel based on manylinux and 2.28 glibc

set -euo pipefail

OUTPUT_DIR="/tmp/kvbm_wheel"
NIXL_REF="0.6.0"
ARCH="amd64"
TARGET="wheel"
KVBM_WHEEL_IMAGE_REF="kvbm-wheel:tmp"

# Flags:
# -o output dir, -n nixl ref, -a arch, -t docker build target, -r image name:tag
while getopts ":o:n:a:t:r:h" opt; do
  case "$opt" in
    o) OUTPUT_DIR="$OPTARG" ;;
    n) NIXL_REF="$OPTARG" ;;
    a) ARCH="$OPTARG" ;;
    t) TARGET="$OPTARG" ;;
    r) KVBM_WHEEL_IMAGE_REF="$OPTARG" ;;
    h)
      echo "Usage: $0 [-o OUTPUT_DIR] [-n NIXL_REF] [-a ARCH{amd64|arm64}] [-t TARGET] [-r IMAGE_NAME:TAG]"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Normalize ARCH and derive ARCH_ALT
case "${ARCH}" in
  amd64|x86_64) ARCH="amd64";  ARCH_ALT="x86_64" ;;
  arm64|aarch64) ARCH="arm64"; ARCH_ALT="aarch64" ;;
  *) echo "Unsupported ARCH: ${ARCH} (use amd64 or arm64)"; exit 1 ;;
esac

BUILD_ARGS=(
  "--build-arg" "NIXL_REF=${NIXL_REF}"
  "--build-arg" "ARCH=${ARCH}"
  "--build-arg" "ARCH_ALT=${ARCH_ALT}"
)

USE_SCCACHE="${USE_SCCACHE:-false}"
SCCACHE_BUCKET="${SCCACHE_BUCKET:-}"
SCCACHE_REGION="${SCCACHE_REGION:-}"

if [[ "${USE_SCCACHE}" == "true" ]]; then
  BUILD_ARGS+=("--build-arg" "USE_SCCACHE=true")
  BUILD_ARGS+=("--build-arg" "SCCACHE_BUCKET=${SCCACHE_BUCKET}")
  BUILD_ARGS+=("--build-arg" "SCCACHE_REGION=${SCCACHE_REGION}")
  BUILD_ARGS+=(
    "--secret" "id=aws-key-id,env=AWS_ACCESS_KEY_ID"
    "--secret" "id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY"
  )
fi

cid=""
cleanup() { [[ -n "${cid}" ]] && docker rm -v "$cid" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker build \
    --target "${TARGET}" \
    "${BUILD_ARGS[@]}" \
    -t "${KVBM_WHEEL_IMAGE_REF}" \
    -f container/Dockerfile.kvbm_wheel .

# Only extract wheels if we built the "wheel" stage
if [[ "${TARGET}" == "wheel" ]]; then
  cid="$(docker create "${KVBM_WHEEL_IMAGE_REF}")"
  mkdir -p "${OUTPUT_DIR}"
  docker cp "${cid}":/opt/dynamo/wheelhouse/. "${OUTPUT_DIR}"/
fi
# trap will remove container
