#!/bin/bash

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


set -euo pipefail

# Validate input parameters
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <DOCKER_REGISTRY> <NAMESPACE> <BENTO_NAME> <BENTO_DIRECTORY>"
  exit 1
fi

DOCKER_REGISTRY=$1
NAMESPACE=$2
BENTO_NAME=$3
BENTO_DIRECTORY=$4

# Check if any of the inputs are empty
if [[ -z "$DOCKER_REGISTRY" || -z "$NAMESPACE" || -z "$BENTO_NAME" || -z "$BENTO_DIRECTORY" ]]; then
  echo "Error: All input parameters (DOCKER_REGISTRY, NAMESPACE, BENTO_NAME, BENTO_DIRECTORY) must be non-empty."
  exit 1
fi

# Check if the specified directory exists
if [ ! -d "$BENTO_DIRECTORY" ]; then
  echo "Error: Directory $BENTO_DIRECTORY does not exist."
  exit 1
fi

echo "Logging into Docker registry: $DOCKER_REGISTRY"
docker login "$DOCKER_REGISTRY"

# Change to the specified directory
cd "$BENTO_DIRECTORY"

# Update the bentofile.yaml with the new BENTO_NAME
echo "Updating bentofile.yaml with name: $BENTO_NAME"
yq eval ".name = \"$BENTO_NAME\"" -i bentofile.yaml

# Build the Bento container
echo "Building Bento image for $BENTO_NAME..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 uv run dynamo build --containerize

# Find the built image
docker_image=$(docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | grep "^$BENTO_NAME:" | sort -r | head -n 1 | awk '{print $1}')
if [[ -z "$docker_image" ]]; then
  echo "Failed to find the built image for $BENTO_NAME"
  exit 1
fi

# Extract the image tag (SHA) from the docker image info
docker_sha=$(echo "$docker_image" | awk -F':' '{print $2}')

echo "Found Docker image: $docker_image"
echo "Docker SHA (bento-version): $docker_sha"

# Tag the image for the registry
docker_tag_for_registry="$DOCKER_REGISTRY/$docker_image"
echo "Tagging image: $docker_tag_for_registry"
docker tag "$docker_image" "$docker_tag_for_registry"

# Push the image
echo "Pushing image: $docker_tag_for_registry"
docker push "$docker_tag_for_registry"

cd -

# Install the Helm chart with the correct bento-version (SHA)
echo "Installing Helm chart with image: $docker_tag_for_registry and bento-version: $docker_sha"
helm install "$BENTO_NAME" ./poc -f ~/bentoml/bentos/"$BENTO_NAME"/"$docker_sha"/bento.yaml --set image="$docker_tag_for_registry" -n "$NAMESPACE"
