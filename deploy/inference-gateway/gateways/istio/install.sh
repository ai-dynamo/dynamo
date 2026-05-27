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

# Install the Gateway API + Gateway API Inference Extension (GAIE) CRDs, Istio
# (with the Gateway API Inference Extension feature flag enabled), and a
# `Gateway` named `inference-gateway` that routes inference traffic via Istio.

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

# Namespace where the Gateway will be deployed.
# Defaults to 'default' if NAMESPACE env var is not set.
NAMESPACE=${NAMESPACE:-default}
ISTIO_NAMESPACE=${ISTIO_NAMESPACE:-istio-system}

# Pinned versions (override via environment if needed).
GATEWAY_API_VERSION=${GATEWAY_API_VERSION:-v1.5.1}
IGW_LATEST_RELEASE=${IGW_LATEST_RELEASE:-v1.2.1}
ISTIO_VERSION=${ISTIO_VERSION:-1.29.2}

echo "Installing inference-gateway (istio) into namespace: $NAMESPACE"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Install the Gateway API.
kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

# Install the Inference Extension CRDs.
kubectl apply \
  -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml"

# Install istioctl if it is not already on PATH.
if ! command -v istioctl >/dev/null 2>&1; then
  echo "istioctl not found on PATH; downloading Istio ${ISTIO_VERSION}..."
  ISTIO_DOWNLOAD_DIR=${ISTIO_DOWNLOAD_DIR:-$(pwd)}
  (
    cd "${ISTIO_DOWNLOAD_DIR}"
    curl -fsSL https://istio.io/downloadIstio | ISTIO_VERSION="${ISTIO_VERSION}" sh -
  )
  export PATH="${ISTIO_DOWNLOAD_DIR}/istio-${ISTIO_VERSION}/bin:$PATH"
fi

# Install Istio with the Gateway API Inference Extension feature flag enabled.
istioctl install -y \
  --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true

# Wait for istiod to be ready before creating the Gateway.
kubectl wait --for=condition=Available --timeout=180s \
  -n "$ISTIO_NAMESPACE" deployment/istiod

kubectl apply -n "$NAMESPACE" -f - <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: istio
  listeners:
    - name: http
      port: 80
      protocol: HTTP
EOF

kubectl wait gateway/inference-gateway -n "$NAMESPACE" \
  --for=condition=Programmed --timeout=180s
