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

# Namespace where the Gateway will be deployed.
# Defaults to 'default' if NAMESPACE env var is not set.
NAMESPACE=${NAMESPACE:-default}
AGW_NAMESPACE=${AGW_NAMESPACE:-agentgateway-system}
echo "Installing inference-gateway into namespace: $NAMESPACE"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Install the Gateway API.
GATEWAY_API_VERSION=v1.5.1
kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

# Install the Inference Extension CRDs.
IGW_LATEST_RELEASE=v1.2.1
kubectl apply \
  -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"

# Install and upgrade agentgateway (includes CRDs).
AGW_VERSION=v1.0.0
helm upgrade -i --create-namespace --namespace "$AGW_NAMESPACE" --version "$AGW_VERSION" \
  agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds

helm upgrade -i --namespace "$AGW_NAMESPACE" --version "$AGW_VERSION" agentgateway \
  oci://cr.agentgateway.dev/charts/agentgateway \
  --set inferenceExtension.enabled=true \
  --wait

kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: agentgateway
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF

kubectl wait gateway/inference-gateway -n "$NAMESPACE" \
  --for=condition=Programmed --timeout=180s
