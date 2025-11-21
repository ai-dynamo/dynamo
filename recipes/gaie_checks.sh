#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -Eeuo pipefail

# ===== Namespace ensure =====
if ! kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
  kubectl create namespace "$NAMESPACE"
fi

KGW_NS="${KGW_NS:-kgateway-system}"

ok()  { printf "✅ %s\n" "$*"; }
fail(){ printf "❌ %s\n" "$*" >&2; exit 1; }
info(){ printf "ℹ️  %s\n" "$*"; }

need() { command -v "$1" >/dev/null 2>&1 || fail "'$1' is required"; }

need kubectl

# ===== Config (env overridable) =====
: "${NAMESPACE:=dynamo}"

# ===== Pre-flight checks =====
command -v helm >/dev/null 2>&1 || { echo "ERROR: helm not found"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found"; exit 1; }

GATEWAY_CRDS=(
  gateways.gateway.networking.k8s.io
  gatewayclasses.gateway.networking.k8s.io
  httproutes.gateway.networking.k8s.io
  referencegrants.gateway.networking.k8s.io
)
info "Checking Gateway API CRDs…"
for c in "${GATEWAY_CRDS[@]}"; do
  kubectl get crd "$c" >/dev/null 2>&1 || fail "Missing CRD: $c (run step a)"
  kubectl wait --for=condition=Established "crd/$c" --timeout=60s >/dev/null || fail "CRD not Established: $c"
done
ok "Gateway API CRDs present & Established"

GAIE_CRDS=(
  inferencemodels.inference.networking.x-k8s.io
  inferencepools.inference.networking.x-k8s.io
)

info "Checking GAIE (Inference Extension) CRDs…"
for c in "${GAIE_CRDS[@]}"; do
  kubectl get crd "$c" >/dev/null 2>&1 || fail "Missing CRD: $c (run step b install of inference extension)"
  kubectl wait --for=condition=Established "crd/$c" --timeout=60s >/dev/null || fail "CRD not Established: $c"
done
ok "GAIE CRDs present & Established"

info "Checking Kgateway controller in namespace '$KGW_NS'…"
# namespace must exist
kubectl get ns "$KGW_NS" >/dev/null 2>&1 || fail "Namespace '$KGW_NS' not found (run step c Helm installs)"

# pods should be running
if ! kubectl get pods -n "$KGW_NS" -l app.kubernetes.io/name=kgateway >/dev/null 2>&1; then
  # fallback label (charts sometimes label differently)
  PODS=$(kubectl get pods -n "$KGW_NS" -o name | grep -E 'kgateway|gateway' || true)
  [[ -z "${PODS:-}" ]] && fail "Kgateway pods not found in '$KGW_NS'"
else
  PODS=$(kubectl get pods -n "$KGW_NS" -l app.kubernetes.io/name=kgateway -o name)
fi
for p in $PODS; do
  kubectl wait -n "$KGW_NS" --for=condition=Ready "$p" --timeout=180s >/dev/null || fail "Pod not Ready: $p"
done
ok "Kgateway controller pods Ready"

kubectl get gateway.gateway.networking.k8s.io inference-gateway -n "$NAMESPACE" >/dev/null 2>&1 || fail "Gateway 'inference-gateway' not found in $NAMESPACE (apply step d manifest)"

ok "GAIE is installed and the gateway is up in namespace '$NAMESPACE'."


