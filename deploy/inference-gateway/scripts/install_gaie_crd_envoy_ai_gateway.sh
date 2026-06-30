#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Installs the Gateway API + Gateway API Inference Extension (GAIE) CRDs
# and Envoy AI Gateway as the underlying Gateway API implementation.
# For the kgateway alternative, see install_gaie_crd_kgateway.sh.

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

# Namespace where the inference-gateway will be deployed
# Defaults to 'default' if NAMESPACE env var is not set
NAMESPACE=${NAMESPACE:-default}
echo "Installing inference-gateway into namespace: $NAMESPACE"

# # Install the Inference Extension CRDs
IGW_LATEST_RELEASE=v1.5.0
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml

# Install Envoy AI Gateway (controller + CRDs)
EG_VERSION="v1.8.1"
EAIG_VERSION="v0.7.0"

helm upgrade -i eg oci://docker.io/envoyproxy/gateway-helm \
  --version "$EG_VERSION" \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/examples/token_ratelimit/envoy-gateway-values-addon.yaml \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/examples/inference-pool/envoy-gateway-values-addon.yaml

helm upgrade -i envoy-ai-gateway-crds oci://docker.io/envoyproxy/ai-gateway-crds-helm \
  --version "$EAIG_VERSION" \
  --namespace envoy-ai-gateway-system \
  --create-namespace

helm upgrade -i envoy-ai-gateway oci://docker.io/envoyproxy/ai-gateway-helm \
  --version "$EAIG_VERSION" \
  --namespace envoy-ai-gateway-system \
  --create-namespace

kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/envoyaigateway/gateway.yaml -n "$NAMESPACE"
