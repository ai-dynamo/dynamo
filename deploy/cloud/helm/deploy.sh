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
export PIPELINES_DOCKER_SERVER="${PIPELINES_DOCKER_SERVER:=${DOCKER_SERVER}}"
export PIPELINES_DOCKER_USERNAME="${PIPELINES_DOCKER_USERNAME:=${DOCKER_USERNAME}}"
export PIPELINES_DOCKER_PASSWORD="${PIPELINES_DOCKER_PASSWORD:=${DOCKER_PASSWORD}}"
export IMAGE_TAG="${IMAGE_TAG:=latest}"  # Default image tag
export DYNAMO_INGRESS_SUFFIX="${DYNAMO_INGRESS_SUFFIX:=dynamo-cloud.com}"
export DOCKER_SECRET_NAME="${DOCKER_SECRET_NAME:=docker-imagepullsecret}"
export INGRESS_ENABLED="${INGRESS_ENABLED:=false}"
export ISTIO_ENABLED="${ISTIO_ENABLED:=false}"
export ISTIO_GATEWAY="${ISTIO_GATEWAY:=istio-system/istio-ingressgateway}"
export INGRESS_CLASS="${INGRESS_CLASS:=nginx}"

# Add command line options
INTERACTIVE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --interactive)
      INTERACTIVE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --interactive       Run in interactive mode"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

if [ "$INTERACTIVE" = true ]; then
  source network-config-wizard.sh
fi


# Check if required variables are set
if [ "$DOCKER_SERVER" = "<your-docker-server>" ]; then
    echo "Error: Please set your DOCKER_SERVER in the script or via environment variable"
    exit 1
fi

# Creates a docker registry secret. Only proceed if both username and password are set
if [[ -n "${DOCKER_USERNAME:-}" && -n "${DOCKER_PASSWORD:-}" ]]; then
  echo "Creating/updating Docker registry secret '$DOCKER_SECRET_NAME' in namespace '$NAMESPACE'..."

  kubectl create secret docker-registry "$DOCKER_SECRET_NAME" \
    --docker-username="$DOCKER_USERNAME" \
    --docker-password="$DOCKER_PASSWORD" \
    --docker-server="$DOCKER_SERVER" \
    --namespace "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
else
  echo "DOCKER_USERNAME and/or DOCKER_PASSWORD not set — skipping docker secret creation."
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
echo "PIPELINES_DOCKER_SERVER: $PIPELINES_DOCKER_SERVER"
echo "PIPELINES_DOCKER_USERNAME: $PIPELINES_DOCKER_USERNAME"
echo "PIPELINES_DOCKER_PASSWORD: [HIDDEN]"
echo "DOCKER_SECRET_NAME: $DOCKER_SECRET_NAME"
echo "INGRESS_ENABLED: $INGRESS_ENABLED"
echo "ISTIO_ENABLED: $ISTIO_ENABLED"
echo "INGRESS_CLASS: $INGRESS_CLASS"
echo "ISTIO_GATEWAY: $ISTIO_GATEWAY"
echo "DYNAMO_INGRESS_SUFFIX: $DYNAMO_INGRESS_SUFFIX"

envsubst '${NAMESPACE} ${RELEASE_NAME} ${DOCKER_USERNAME} ${DOCKER_PASSWORD} ${DOCKER_SERVER} ${IMAGE_TAG} ${DYNAMO_INGRESS_SUFFIX} ${PIPELINES_DOCKER_SERVER} ${PIPELINES_DOCKER_USERNAME} ${PIPELINES_DOCKER_PASSWORD} ${DOCKER_SECRET_NAME} ${INGRESS_ENABLED} ${ISTIO_ENABLED} ${INGRESS_CLASS} ${ISTIO_GATEWAY}' < dynamo-platform-values.yaml > generated-values.yaml
echo "generated file contents:"
cat generated-values.yaml

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
