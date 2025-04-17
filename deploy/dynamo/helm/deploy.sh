#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Use system helm
HELM_CMD=$(which helm)

# Set default values only if not already set
export NAMESPACE="${NAMESPACE:=dynamo-cloud}"  # Default namespace
export RELEASE_NAME="${RELEASE_NAME:=${NAMESPACE}}"  # Default release name is same as namespace
export DOCKER_USERNAME="${DOCKER_USERNAME:=<your-docker-username>}"  # Default docker username
export DOCKER_PASSWORD="${DOCKER_PASSWORD:=<your-docker-password>}"  # Default docker password
export DOCKER_SERVER="${DOCKER_SERVER:=<your-docker-server>}"  # Default docker server
export IMAGE_TAG="${IMAGE_TAG:=latest}"  # Default image tag
export DYNAMO_INGRESS_SUFFIX="${DYNAMO_INGRESS_SUFFIX:=dynamo-cloud.com}"

# Check if required variables are set
if [ "$DOCKER_USERNAME" = "<your-docker-username>" ]; then
    echo "Error: Please set your DOCKER_USERNAME in the script or via environment variable"
    exit 1
fi

if [ "$DOCKER_PASSWORD" = "<your-docker-password>" ]; then
    echo "Error: Please set your DOCKER_PASSWORD in the script or via environment variable"
    exit 1
fi

if [ "$DOCKER_SERVER" = "<your-docker-server>" ]; then
    echo "Error: Please set your DOCKER_SERVER in the script or via environment variable"
    exit 1
fi

# Function to retry commands
retry_command() {
    local -r cmd="$1"
    local -r max_attempts=${2:-3}
    local -r delay=${3:-5}
    local attempt=1

    until eval "$cmd"; do
        if ((attempt >= max_attempts)); then
            echo "Command '$cmd' failed after $attempt attempts"
            return 1
        fi
        echo "Command '$cmd' failed, attempt $attempt of $max_attempts. Retrying in ${delay}s..."
        ((attempt++))
        sleep "$delay"
    done
}

# Update the helm repo and build the dependencies
retry_command "$HELM_CMD repo add nats https://nats-io.github.io/k8s/helm/charts/" 5 5 && \
retry_command "$HELM_CMD repo add bitnami https://charts.bitnami.com/bitnami" 5 5 && \
retry_command "$HELM_CMD repo add minio https://charts.min.io/" 5 5 && \
retry_command "$HELM_CMD repo update" 5 5

cd platform
cd components/operator
retry_command "$HELM_CMD dependency update" 5 5
cd ../..
cd components/api-store
retry_command "$HELM_CMD dependency update" 5 5
cd ../..
retry_command "$HELM_CMD dep update" 7 5
cd ..

# Generate the values file
echo "Generating values file with:"
echo "NAMESPACE: $NAMESPACE"
echo "RELEASE_NAME: $RELEASE_NAME"
echo "IMAGE_TAG: $IMAGE_TAG"
echo "DOCKER_USERNAME: $DOCKER_USERNAME"
echo "DOCKER_SERVER: $DOCKER_SERVER"
echo "DOCKER_PASSWORD: [HIDDEN]"

echo "generated file contents:"
envsubst '${NAMESPACE} ${RELEASE_NAME} ${DOCKER_USERNAME} ${DOCKER_PASSWORD} ${DOCKER_SERVER} ${IMAGE_TAG} ${DYNAMO_INGRESS_SUFFIX}' < dynamo-platform-values.yaml

envsubst '${NAMESPACE} ${RELEASE_NAME} ${DOCKER_USERNAME} ${DOCKER_PASSWORD} ${DOCKER_SERVER} ${IMAGE_TAG} ${DYNAMO_INGRESS_SUFFIX}' < dynamo-platform-values.yaml > generated-values.yaml

echo ""
echo "Generated values file saved as generated-values.yaml"

# Build dependencies before installation
echo "Building helm dependencies..."
cd platform
retry_command "$HELM_CMD dep build" 5 5
cd ..

# Install/upgrade the helm chart
echo "Installing/upgrading helm chart..."
$HELM_CMD upgrade --install $RELEASE_NAME platform/ \
  -f generated-values.yaml \
  --create-namespace \
  --namespace ${NAMESPACE}

echo "Helm chart deployment complete"
