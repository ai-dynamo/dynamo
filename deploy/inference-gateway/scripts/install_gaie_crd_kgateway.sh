#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

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

kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/kgateway/gateway.yaml -n "$NAMESPACE"

kubectl patch gateway inference-gateway -n "$NAMESPACE" --type='json' \
  -p='[{"op": "replace", "path": "/spec/gatewayClassName", "value": "kgateway"}]'
