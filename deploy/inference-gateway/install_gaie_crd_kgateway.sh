#!/usr/bin/env bash

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

#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR

MODEL_NAMESPACE=${MODEL_NAMESPACE:-my-model}
KGATEWAY_SYSTEM_NAMESPACE=${KGATEWAY_SYSTEM_NAMESPACE:-kgateway-system}

GATEWAY_API_VERSION=${GATEWAY_API_VERSION:-v1.3.0}
INFERENCE_EXTENSION_VERSION=${INFERENCE_EXTENSION_VERSION:-v0.5.1}
KGATEWAY_VERSION=${KGATEWAY_VERSION:-v2.0.3}

GATEWAY_API_MANIFEST="https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"
GAIE_MANIFEST="https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${INFERENCE_EXTENSION_VERSION}/manifests.yaml"
KGATEWAY_CRDS_CHART="oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds"
KGATEWAY_CHART="oci://cr.kgateway.dev/kgateway-dev/charts/kgateway"


GATEWAY_INSTANCE_MANIFEST="https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/manifests/gateway/kgateway/gateway.yaml"

# Baseline marker
BASELINE_CM_NAMESPACE=kube-system
BASELINE_CM_NAME=gaie-kgateway-baseline
BASELINE_KEY=versions
BASELINE_VAL="gateway_api=${GATEWAY_API_VERSION},gaie=${INFERENCE_EXTENSION_VERSION},kgateway=${KGATEWAY_VERSION}"


ns()       { kubectl get ns "$1" >/dev/null 2>&1 || kubectl create ns "$1"; }
have_crd() { kubectl get crd "$1" >/dev/null 2>&1; }

cm_matches() {
  kubectl get configmap "$BASELINE_CM_NAME" -n "$BASELINE_CM_NAMESPACE" -o jsonpath='{.data.'"$BASELINE_KEY"'}' 2>/dev/null | grep -qxF "$BASELINE_VAL"
}

set_cm() {
  kubectl -n "$BASELINE_CM_NAMESPACE" create configmap "$BASELINE_CM_NAME" \
    --from-literal="$BASELINE_KEY=$BASELINE_VAL" \
    --dry-run=client -o yaml | kubectl apply -f -
}

helm_chart_version() {
  # Prints deployed chart version (e.g., 2.0.3) or empty if not installed
  local rel ns json
  rel="$1"; ns="$2"
  json=$(helm -n "$ns" ls -f "^${rel}$" -o json 2>/dev/null || true)
  if [[ -z "$json" || "$json" == "[]" ]]; then
    echo ""
  else
    # .[0].chart looks like "kgateway-2.0.3" → cut the suffix after last dash
    echo "$json" | jq -r '.[0].chart' | awk -F- '{print $NF}'
  fi
}


ns "$BASELINE_CM_NAMESPACE"
if cm_matches; then
  echo "Baseline marker already set: $BASELINE_VAL"
else
  echo "Setting/Updating baseline marker: $BASELINE_VAL"
  set_cm
fi


ns "$MODEL_NAMESPACE"
ns "$KGATEWAY_SYSTEM_NAMESPACE"

# Install Gateway API (cluster-scoped)
if have_crd gateways.gateway.networking.k8s.io; then
  echo "Gateway API CRDs already present; skipping install."
else
  echo "Installing Gateway API ${GATEWAY_API_VERSION}..."
  kubectl apply -f "$GATEWAY_API_MANIFEST"
  kubectl wait --for=condition=Established crd/gateways.gateway.networking.k8s.io --timeout=120s
fi

# Install GAIE CRDs (cluster-scoped)
if have_crd inferenceclasses.gateway.networking.k8s.io; then
  echo "GAIE CRDs already present; skipping install."
else
  echo "Installing GAIE CRDs ${INFERENCE_EXTENSION_VERSION}..."
  kubectl apply -f "$GAIE_MANIFEST"
  kubectl wait --for=condition=Established crd/inferenceclasses.gateway.networking.k8s.io --timeout=120s || true
fi

# Install kGateway (cluster-scoped controller + CRDs)
# Only upgrade if the chart version differs or not installed yet.
current_crds_ver=$(helm_chart_version kgateway-crds "$KGATEWAY_SYSTEM_NAMESPACE")
current_ctrl_ver=$(helm_chart_version kgateway "$KGATEWAY_SYSTEM_NAMESPACE")

if [[ "$current_crds_ver" != "$KGATEWAY_VERSION" ]]; then
  echo "Installing/Upgrading kGateway CRDs to ${KGATEWAY_VERSION} (was: ${current_crds_ver:-none})..."
  helm upgrade -i --create-namespace --namespace "$KGATEWAY_SYSTEM_NAMESPACE" \
    --version "$KGATEWAY_VERSION" kgateway-crds "$KGATEWAY_CRDS_CHART"
else
  echo "kGateway CRDs already at ${KGATEWAY_VERSION}; skipping."
fi

if [[ "$current_ctrl_ver" != "$KGATEWAY_VERSION" ]]; then
  echo "Installing/Upgrading kGateway controller to ${KGATEWAY_VERSION} (was: ${current_ctrl_ver:-none})..."
  helm upgrade -i --namespace "$KGATEWAY_SYSTEM_NAMESPACE" \
    --version "$KGATEWAY_VERSION" kgateway "$KGATEWAY_CHART" \
    --set inferenceExtension.enabled=true
else
  echo "kGateway controller already at ${KGATEWAY_VERSION}; skipping."
fi

# Install Gateway instance (namespaced)
if kubectl get gateway inference-gateway -n "$MODEL_NAMESPACE" >/dev/null 2>&1; then
  echo "Gateway instance 'inference-gateway' already exists in ${MODEL_NAMESPACE}; skipping apply."
else
  echo "Creating Gateway instance in ${MODEL_NAMESPACE}..."
  kubectl apply -f "$GATEWAY_INSTANCE_MANIFEST" -n "$MODEL_NAMESPACE"
fi

echo "Done. Cluster-wide components only changed when needed."

