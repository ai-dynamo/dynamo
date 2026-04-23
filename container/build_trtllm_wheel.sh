#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Build the TRT-LLM wheel.

# This script builds the TRT-LLM base image for Dynamo with TensorRT-LLM.

while getopts "c:o:a:n:u:" opt; do
  case ${opt} in
    c) TRTLLM_COMMIT=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    a) ARCH=$OPTARG ;;
    n) NIXL_COMMIT=$OPTARG ;;
    u) TRTLLM_GIT_URL=$OPTARG ;;
    *) echo "Usage: $(basename $0) [-c commit] [-o output_dir] [-a arch] [-n nixl_commit] [-u git_url]"
       echo "  -c: TensorRT-LLM commit to build"
       echo "  -o: Output directory for wheel files"
       echo "  -a: Architecture (amd64 or arm64)"
       echo "  -n: NIXL commit"
       echo "  -u: TensorRT-LLM git URL"
       exit 1 ;;
  esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/tmp/trtllm_wheel"
fi

# Set default TensorRT-LLM git URL if not specified.
# Default points at the galletas1712 fork because the Kimi K2.5 + Eagle3
# shadow-failover proof pins the runtime image to a fork-only commit
# (see container/context.yaml::trtllm.github_trtllm_commit).
if [ -z "$TRTLLM_GIT_URL" ]; then
    TRTLLM_GIT_URL="https://github.com/galletas1712/TensorRT-LLM.git"
fi

# Store directory where script is being launched from
MAIN_DIR=$(dirname "$(readlink -f "$0")")

(cd /tmp && \
# Clone the TensorRT-LLM repository.
# Use a bare fetch-and-checkout flow so that ${TRTLLM_COMMIT} does not need
# to be reachable from ${TRTLLM_GIT_URL}'s default branch. The Kimi K2.5 +
# Eagle3 proof pins a SHA that lives only on a feature branch of the fork
# (see container/context.yaml::trtllm.github_trtllm_commit), so the previous
# 'checkout main; pull; checkout <sha>' flow would fail on any SHA that is
# not yet merged to main on the selected remote.
if [ ! -d "TensorRT-LLM" ]; then
  git init TensorRT-LLM
  (cd TensorRT-LLM && git remote add origin "${TRTLLM_GIT_URL}")
fi

cd TensorRT-LLM

# Make sure the remote URL matches the requested URL in case the directory was
# reused from a previous build that targeted a different fork.
git remote set-url origin "${TRTLLM_GIT_URL}"

# Fetch the exact commit directly. This works for any reachable SHA on the
# remote (branches, tags, or PR heads) without having to enumerate refs or
# pull the entire history of main first.
git fetch --no-tags --prune --depth=1 origin "${TRTLLM_COMMIT}"
git checkout --detach FETCH_HEAD

# Update the submodules.
git submodule update --init --recursive
git lfs pull

VERSION_FILE="tensorrt_llm/version.py"

# Check if file exists
if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: $VERSION_FILE not found"
    exit 1
fi

# Create a backup of the original version file
cp $VERSION_FILE ${VERSION_FILE}.bak

# Check if version line exists
if ! grep -q "^__version__" "$VERSION_FILE"; then
    echo "Error: __version__ not found in $VERSION_FILE"
    exit 1
fi

# Append suffix to version
COMMIT_VERSION=$(git rev-parse --short HEAD)
sed -i "s/__version__ = \"\(.*\)\"/__version__ = \"\1+dev${COMMIT_VERSION}\"/" "$VERSION_FILE"

echo "Updated version:"
grep "__version__" "$VERSION_FILE"

# Maximize parallelism for the wheel build. TRTLLM's wheel build_wheel.py
# reads JOBS from the environment, and cmake/make rules accept MAKEFLAGS.
# The CI builders have plenty of CPU, so push -j high when the caller has
# not already set one.
if [ -z "${JOBS:-}" ]; then
    export JOBS="$(nproc 2>/dev/null || echo 16)"
fi
if [ -z "${MAKEFLAGS:-}" ]; then
    export MAKEFLAGS="-j${JOBS}"
fi

if [ "$ARCH" = "amd64" ]; then
    # Need to build in the Triton Devel Image for NIXL support.
    make -C docker tritondevel_build
    make -C docker wheel_build DEVEL_IMAGE=tritondevel BUILD_WHEEL_OPTS="--extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl --job_count ${JOBS}"
else
    # NIXL backend is not supported on arm64 for TensorRT-LLM.
    # See here: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/common/install_nixl.sh
    make -C docker wheel_build BUILD_WHEEL_OPTS="--job_count ${JOBS}"
fi

# Copy the wheel to the host
mkdir -p $OUTPUT_DIR

docker create --name trtllm_wheel_container docker.io/tensorrt_llm/wheel:latest
docker cp trtllm_wheel_container:/src/tensorrt_llm/build $OUTPUT_DIR/
cp $OUTPUT_DIR/build/*.whl $OUTPUT_DIR/
docker rm trtllm_wheel_container || true

# Restore the original version file
mv ${VERSION_FILE}.bak $VERSION_FILE

)

# Store the commit hash in the output directory to ensure the wheel is built from the correct commit.
rm -rf $OUTPUT_DIR/commit.txt
echo ${ARCH}_${TRTLLM_COMMIT} > $OUTPUT_DIR/commit.txt

echo "TRT-LLM wheel built successfully."
ls -al $OUTPUT_DIR
