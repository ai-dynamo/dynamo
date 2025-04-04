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

# Set default values only if not already set
export NAMESPACE="${NAMESPACE:=cai-system}"  # Default namespace
export NGC_TOKEN="${NGC_TOKEN:=<your-ngc-token>}"  # Default NGC token
export CI_COMMIT_SHA="${CI_COMMIT_SHA:=250e2e0f93f7af3d83a4a0ff992e56956f7651f2}"  # Default commit SHA
export RELEASE_NAME="${RELEASE_NAME:=dynamo-platform}"  # Default commit SHA
export DYNAMO_INGRESS_SUFFIX="${DYNAMO_INGRESS_SUFFIX:=}"


# Check if required variables are set
if [ "$NGC_TOKEN" = "<your-ngc-token>" ]; then
    echo "Error: Please set your NGC_TOKEN in the script or via environment variable"
    exit 1
fi

# Update the helm repo and build the dependencies
cd platform
cd components/operator
helm dependency update
cd ../..
cd components/api-server
helm dependency update
cd ../..
helm dep update
helm repo update
cd ..

# Generate the values file
echo "Generating values file with:"
echo "NAMESPACE: $NAMESPACE"
echo "CI_COMMIT_SHA: $CI_COMMIT_SHA"
echo "NGC_TOKEN: [HIDDEN]"
echo "RELEASE_NAME: $RELEASE_NAME"

echo "generated file contents:"
envsubst '${NAMESPACE} ${NGC_TOKEN} ${CI_COMMIT_SHA} ${RELEASE_NAME} ${DYNAMO_INGRESS_SUFFIX}' < dynamo-platform-values.yaml

envsubst '${NAMESPACE} ${NGC_TOKEN} ${CI_COMMIT_SHA} ${RELEASE_NAME} ${DYNAMO_INGRESS_SUFFIX}' < dynamo-platform-values.yaml > generated-values.yaml

echo "Generated values file saved as generated-values.yaml"


# Install/upgrade the helm chart
echo "Installing/upgrading helm chart..."
helm upgrade --install $RELEASE_NAME platform/ \
  -f generated-values.yaml \
  --create-namespace \
  --namespace ${NAMESPACE}

echo "Helm chart deployment complete"
