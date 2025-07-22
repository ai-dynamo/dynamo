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
trap 'echo "Error at line $LINENO. Exiting."' ERR

required_tools=(envsubst kubectl helm)

for tool in "${required_tools[@]}"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: Required tool '$tool' is not installed or not in PATH."
    exit 1
  fi
done


# Use system helm
HELM_CMD=$(which helm)

# Set default values only if not already set
export NAMESPACE="${NAMESPACE:=dynamo-cloud}"  # Default namespace
export RELEASE_NAME="${RELEASE_NAME:=${NAMESPACE}}"  # Default release name is same as namespace
export DOCKER_USERNAME="${DOCKER_USERNAME:-}"  # Default docker username is empty
export DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"  # Default docker password is empty
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
export VIRTUAL_SERVICE_SUPPORTS_HTTPS="${VIRTUAL_SERVICE_SUPPORTS_HTTPS:=false}"
export ENABLE_LWS="${ENABLE_LWS:=false}"

# Add command line options
INTERACTIVE=false
INSTALL_CRDS=false
YAML_ONLY=false
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --interactive)
      INTERACTIVE=true
      shift
      ;;
    --crds)
      INSTALL_CRDS=true
      shift
      ;;
    --yaml-only)
      YAML_ONLY=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --interactive       Run in interactive mode"
      echo "  --yaml-only         Only generate generated-values.yaml and exit"
      echo "  --help              Show this help message"
      echo "  --crds              Also install the CRDs"
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

  # Transform docker.io URLs to index.docker.io/v1/
  DOCKER_SERVER_FOR_SECRET="$DOCKER_SERVER"
  if [[ "$DOCKER_SERVER" == "docker.io" || "$DOCKER_SERVER" == "docker.io/"* ]]; then
    DOCKER_SERVER_FOR_SECRET="https://index.docker.io/v1/"
  fi

  kubectl create secret docker-registry "$DOCKER_SECRET_NAME" \
    --docker-username="$DOCKER_USERNAME" \
    --docker-password="$DOCKER_PASSWORD" \
    --docker-server="$DOCKER_SERVER_FOR_SECRET" \
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
# retry_command "$HELM_CMD repo add bitnami https://charts.bitnami.com/bitnami" 5 5 && \
# retry_command "$HELM_CMD repo add minio https://charts.min.io/" 5 5 && \
retry_command "$HELM_CMD repo update" 5 5


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
echo "VIRTUAL_SERVICE_SUPPORTS_HTTPS: $VIRTUAL_SERVICE_SUPPORTS_HTTPS"
echo "INSTALL_CRDS: $INSTALL_CRDS"

envsubst '${NAMESPACE} ${RELEASE_NAME} ${DOCKER_USERNAME} ${DOCKER_PASSWORD} ${DOCKER_SERVER} ${IMAGE_TAG} ${DYNAMO_INGRESS_SUFFIX} ${PIPELINES_DOCKER_SERVER} ${PIPELINES_DOCKER_USERNAME} ${PIPELINES_DOCKER_PASSWORD} ${DOCKER_SECRET_NAME} ${INGRESS_ENABLED} ${ISTIO_ENABLED} ${INGRESS_CLASS} ${ISTIO_GATEWAY} ${VIRTUAL_SERVICE_SUPPORTS_HTTPS} ${ENABLE_LWS}' < dynamo-platform-values.yaml > generated-values.yaml
echo "generated file contents:"
cat generated-values.yaml

echo ""
echo "Generated values file saved as generated-values.yaml"
if [ "$YAML_ONLY" = true ]; then
  echo "--yaml-only flag detected; skipping Helm deployment steps."
  exit 0
fi

# Build dependencies before installation
echo "Building helm dependencies..."
cd platform
retry_command "$HELM_CMD dep build" 5 5
cd ..

# Install/upgrade the helm chart for the CRDs
if [ "$INSTALL_CRDS" = true ]; then
  echo "Installing/upgrading helm chart for the CRDs..."
  $HELM_CMD upgrade --install dynamo-crds crds/ --namespace default --wait --atomic
fi

# Build Platform
echo "Building platform..."
$HELM_CMD dep build ./platform/

# Install platform
echo "Installing platform..."
helm upgrade --install dynamo-platform ./platform/ \
  --namespace ${NAMESPACE} \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
echo "Helm chart deployment complete"
