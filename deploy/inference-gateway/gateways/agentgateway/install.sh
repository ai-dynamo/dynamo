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

# Install the Gateway API + Gateway API Inference Extension (GAIE) CRDs, the
# agentgateway control plane, and a `Gateway` named `inference-gateway` that
# routes inference traffic via agentgateway.

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

# Namespace where the Gateway will be deployed.
# Defaults to 'default' if NAMESPACE env var is not set.
NAMESPACE=${NAMESPACE:-default}
AGW_NAMESPACE=${AGW_NAMESPACE:-agentgateway-system}

# Pinned versions (override via environment if needed).
GATEWAY_API_VERSION=${GATEWAY_API_VERSION:-v1.5.1}
IGW_LATEST_RELEASE=${IGW_LATEST_RELEASE:-v1.2.1}
AGW_VERSION=${AGW_VERSION:-v1.0.0}

echo "Installing inference-gateway (agentgateway) into namespace: $NAMESPACE"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Install the Gateway API.
kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

# Install the Inference Extension CRDs.
kubectl apply \
  -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"

# Install and upgrade agentgateway (includes CRDs).
helm upgrade -i --create-namespace --namespace "$AGW_NAMESPACE" --version "$AGW_VERSION" \
  agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds

helm upgrade -i --namespace "$AGW_NAMESPACE" --version "$AGW_VERSION" agentgateway \
  oci://cr.agentgateway.dev/charts/agentgateway \
  --set inferenceExtension.enabled=true \
  --wait

# Create an AgentgatewayParameters resource that excludes Istio sidecar injection
# from the agentgateway-proxy pods. When the deployment namespace has
# istio-injection=enabled, the Istio sidecar intercepts the ext_proc gRPC
# connection from agentgateway-proxy to EPP (port 9002), causing all inference
# requests to return HTTP 500. Setting sidecar.istio.io/inject: "false" on the
# pod template prevents sidecar injection so that ext_proc traffic reaches EPP
# directly. This annotation is a no-op on clusters where Istio is not installed.
#
# AgentgatewayParameters must live in the same namespace as the Gateway because
# Gateway API's infrastructure.parametersRef is a LocalParametersReference
# (no namespace field).
kubectl apply --server-side -n "$NAMESPACE" -f - <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayParameters
metadata:
  name: inference-gateway-params
spec:
  deployment:
    spec:
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
EOF

kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: agentgateway
  infrastructure:
    parametersRef:
      group: agentgateway.dev
      kind: AgentgatewayParameters
      name: inference-gateway-params
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF

kubectl wait gateway/inference-gateway -n "$NAMESPACE" \
  --for=condition=Programmed --timeout=180s
