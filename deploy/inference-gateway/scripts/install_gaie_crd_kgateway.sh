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

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

# Install the Gateway API
GATEWAY_API_VERSION=v1.4.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml


# Install the Inference Extension CRDs
IGW_LATEST_RELEASE=v1.2.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml


# Install and upgrade Kgateway (includes CRDs)
KGTW_VERSION=v2.1.1
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION \
  kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway \
  oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
  --set inferenceExtension.enabled=true

kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/kgateway/gateway.yaml

kubectl patch gateway inference-gateway --type='json' \
  -p='[{"op": "replace", "path": "/spec/gatewayClassName", "value": "kgateway"}]'
