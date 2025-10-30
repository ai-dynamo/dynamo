#!/usr/bin/env bash
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
KVBM_WHEEL_IMAGE_REF="kvbm-wheel:tmp"

# Flags:
# -o output dir, -r image name:tag
while getopts ":o:r:h" opt; do
  case "$opt" in
    o) OUTPUT_DIR="$OPTARG" ;;
    r) KVBM_WHEEL_IMAGE_REF="$OPTARG" ;;
    h)
      echo "Usage: $0 [-o OUTPUT_DIR] [-r IMAGE:TAG]"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_ARGS=( --framework none --target dev --tag "$KVBM_WHEEL_IMAGE_REF" --enable-kvbm )

"$SCRIPT_DIR/build.sh" "${BUILD_ARGS[@]}"

mkdir -p "$OUTPUT_DIR"
docker run --rm \
  --entrypoint /bin/bash \
  -v "$OUTPUT_DIR:/out" \
  "$KVBM_WHEEL_IMAGE_REF" \
  -lc 'cp /opt/dynamo/wheelhouse/kvbm-*.whl /out/'
