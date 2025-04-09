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
export NAMESPACE="${NAMESPACE:=cai-system}"  # Default namespace
export NGC_TOKEN="${NGC_TOKEN:=<your-ngc-token>}"  # Default NGC token
export CI_REGISTRY_IMAGE="${CI_REGISTRY_IMAGE:=<your-registry>/<your-org>}"  # Default registry/org
export CI_COMMIT_SHA="${CI_COMMIT_SHA:=250e2e0f93f7af3d83a4a0ff992e56956f7651f2}"  # Default commit SHA
export RELEASE_NAME="${RELEASE_NAME:=dynamo-platform}"  # Default release name
export DYNAMO_INGRESS_SUFFIX="${DYNAMO_INGRESS_SUFFIX:=}"

# Check if required variables are set
if [ "$NGC_TOKEN" = "<your-ngc-token>" ]; then
    echo "Error: Please set your NGC_TOKEN in the script or via environment variable"
    exit 1
fi

if [ "$CI_REGISTRY_IMAGE" = "<your-registry>/<your-org>" ]; then
    echo "Error: Please set your CI_REGISTRY_IMAGE in the script or via environment variable"
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
echo "CI_COMMIT_SHA: $CI_COMMIT_SHA"
echo "CI_REGISTRY_IMAGE: $CI_REGISTRY_IMAGE"
echo "NGC_TOKEN: [HIDDEN]"
echo "RELEASE_NAME: $RELEASE_NAME"

echo "generated file contents:"
envsubst '${NAMESPACE} ${NGC_TOKEN} ${CI_COMMIT_SHA} ${RELEASE_NAME} ${DYNAMO_INGRESS_SUFFIX} ${CI_REGISTRY_IMAGE}' < dynamo-platform-values.yaml

envsubst '${NAMESPACE} ${NGC_TOKEN} ${CI_COMMIT_SHA} ${RELEASE_NAME} ${DYNAMO_INGRESS_SUFFIX} ${CI_REGISTRY_IMAGE}' < dynamo-platform-values.yaml > generated-values.yaml

echo ""
echo "Generated values file saved as generated-values.yaml"

Build dependencies before installation
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
