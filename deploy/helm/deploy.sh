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
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <DOCKER_REGISTRY> <NAMESPACE> <DYNAMO_DIRECTORY> <DYNAMO_IDENTIFIER> [<DYNAMO_CONFIG_FILE>]"
  echo "Note: DYNAMO_CONFIG_FILE is optional"
  exit 1
fi

DOCKER_REGISTRY=$1
NAMESPACE=$2
DYNAMO_DIRECTORY=$3
DYNAMO_IDENTIFIER=$4
DYNAMO_CONFIG_FILE=${5:-""}  # Set to empty string if not provided

# Check if any of the inputs are empty
if [[ -z "$DOCKER_REGISTRY" || -z "$NAMESPACE" || -z "$DYNAMO_IDENTIFIER" || -z "$DYNAMO_DIRECTORY" ]]; then
  echo "Error: All input parameters (DOCKER_REGISTRY, NAMESPACE, DYNAMO_IDENTIFIER, DYNAMO_DIRECTORY) must be non-empty."
  exit 1
fi

# Check if the specified directory exists
if [ ! -d "$DYNAMO_DIRECTORY" ]; then
  echo "Error: Directory $DYNAMO_DIRECTORY does not exist."
  exit 1
fi

echo "Logging into Docker registry: $DOCKER_REGISTRY"
docker login "$DOCKER_REGISTRY"

# Change to the specified directory
cd "$DYNAMO_DIRECTORY"

# Build the Bento container
echo "Building Bento image for $DYNAMO_IDENTIFIER..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 uv run dynamo build --containerize $DYNAMO_IDENTIFIER

# Extract the module and the bento name
DYNAMO_MODULE=$(echo "$DYNAMO_IDENTIFIER" | awk -F':' '{print $1}' | tr '[:upper:]' '[:lower:]')
DYNAMO_NAME=$(echo "$DYNAMO_IDENTIFIER" | awk -F':' '{print $2}' | tr '[:upper:]' '[:lower:]')


# Find the built image
docker_image=$(docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | grep "^$DYNAMO_NAME:" | sort -r -k2,3 | head -n 1 | awk '{print $1}')
if [[ -z "$docker_image" ]]; then
  echo "Failed to find the built image for $DYNAMO_NAME"
  exit 1
fi

# Extract the image tag (SHA) from the docker image info
docker_sha=$(echo "$docker_image" | awk -F':' '{print $2}')

echo "Found Docker image: $docker_image"
echo "Docker SHA: $docker_sha"

# Tag the image for the registry
docker_tag_for_registry="$DOCKER_REGISTRY/$docker_image"
echo "Tagging image: $docker_tag_for_registry"
docker tag "$docker_image" "$docker_tag_for_registry"

# Push the image
echo "Pushing image: $docker_tag_for_registry"
docker push "$docker_tag_for_registry"

cd -

# Install the Helm chart with the correct tag (SHA)
echo "Installing Helm chart with image: $docker_tag_for_registry"
HELM_RELEASE="${DYNAMO_MODULE//_/\-}"
helm upgrade -i "$HELM_RELEASE" ./chart -f ~/bentoml/bentos/"$DYNAMO_NAME"/"$docker_sha"/bento.yaml --set image="$docker_tag_for_registry" --set dynamoIdentifier="$DYNAMO_IDENTIFIER" --set configFilePath="$DYNAMO_CONFIG_FILE" -n "$NAMESPACE"