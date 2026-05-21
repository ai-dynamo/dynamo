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

# Namespace where the inference-gateway will be deployed
# Defaults to 'default' if NAMESPACE env var is not set
NAMESPACE=${NAMESPACE:-default}
echo "Installing inference-gateway into namespace: $NAMESPACE"

# Install the Gateway API
GATEWAY_API_VERSION=v1.4.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml


# Install the Inference Extension CRDs
IGW_LATEST_RELEASE=v1.5.0-rc.2
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml


# Install and upgrade Kgateway (includes CRDs)
KGTW_VERSION=v2.1.1
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION \
  kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway \
  oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
  --set inferenceExtension.enabled=true

# Create a GatewayParameters resource that excludes Istio sidecar injection from the
# kgateway-proxy pods. When the deployment namespace has istio-injection=enabled, the
# Istio sidecar intercepts the ext_proc gRPC connection from kgateway-proxy to EPP
# (port 9002), causing all inference requests to return HTTP 500. Setting
# sidecar.istio.io/inject: "false" prevents sidecar injection on the proxy pod so
# that ext_proc traffic reaches EPP directly. This annotation is a no-op on clusters
# where Istio is not installed.
#
# GatewayParameters must live in the same namespace as the Gateway because
# Gateway API's infrastructure.parametersRef is a LocalParametersReference
# (no namespace field).
kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: gateway.kgateway.dev/v1alpha1
kind: GatewayParameters
metadata:
  name: inference-gateway-params
spec:
  kube:
    podTemplate:
      extraAnnotations:
        sidecar.istio.io/inject: "false"
EOF

kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: kgateway
  infrastructure:
    parametersRef:
      group: gateway.kgateway.dev
      kind: GatewayParameters
      name: inference-gateway-params
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF
