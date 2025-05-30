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

# Build the TRT-LLM wheel.

# This script builds the TRT-LLM base image for Dynamo with TensorRT-LLM.

while getopts "c:o:a:" opt; do
  case ${opt} in
    c) TRTLLM_COMMIT=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    a) ARCH=$OPTARG ;;
    *) echo "Usage: $(basename $0) [-c commit] [-o output_dir] [-a arch]"
       echo "  -c: TensorRT-LLM commit to build"
       echo "  -o: Output directory for wheel files"
       echo "  -a: Architecture (amd64 or arm64)"
       exit 1 ;;
  esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/tmp/trtllm_wheel"
fi


(cd /tmp && \
# Clone the TensorRT-LLM repository.
if [ ! -d "TensorRT-LLM" ]; then
  git clone --single-branch --branch main https://github.com/NVIDIA/TensorRT-LLM.git
fi

cd TensorRT-LLM

# Checkout the specified commit.
# Switch to the main branch to pull the latest changes.
git checkout main
git pull
git checkout $TRTLLM_COMMIT

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


make -C docker wheel_build

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