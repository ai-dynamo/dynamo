#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -Eeuo pipefail

# ===== Namespace ensure =====
# The documented GAIE flow sets NAMESPACE explicitly; the fallback matches the installer.
: "${NAMESPACE:=default}"

GATEWAY_IMPL="${1:-}"
case "${GATEWAY_IMPL}" in
  agentgateway)
    : "${GATEWAY_CONTROLLER_NAMESPACE:=${AGW_NAMESPACE:-agentgateway-system}}"
    : "${GATEWAY_LABEL_FILTER:=app.kubernetes.io/name=agentgateway}"
    ;;
  envoy-ai-gateway)
    : "${GATEWAY_CONTROLLER_NAMESPACE:=${EAIGW_NAMESPACE:-envoy-gateway-system}}"
    : "${GATEWAY_LABEL_FILTER:=app.kubernetes.io/instance=eg}"
    ;;
  "")
    echo "Usage: $0 <agentgateway|envoy-ai-gateway>" >&2
    exit 1
    ;;
  *)
    echo "ERROR: unknown gateway implementation '${GATEWAY_IMPL}'. Must be 'agentgateway' or 'envoy-ai-gateway'." >&2
    exit 1
    ;;
esac

ok()  { printf "✅ %s\n" "$*"; }
fail(){ printf "❌ %s\n" "$*" >&2; exit 1; }
info(){ printf "ℹ️  %s\n" "$*"; }

need() { command -v "$1" >/dev/null 2>&1 || fail "'$1' is required"; }

need kubectl

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
  inferencemodelrewrites.inference.networking.x-k8s.io
  inferenceobjectives.inference.networking.x-k8s.io
  inferencepoolimports.inference.networking.x-k8s.io
  inferencepools.inference.networking.k8s.io
)

info "Checking GAIE (Inference Extension) CRDs…"
for c in "${GAIE_CRDS[@]}"; do
  kubectl get crd "$c" >/dev/null 2>&1 || fail "Missing CRD: $c (run step b install of inference extension)"
  kubectl wait --for=condition=Established "crd/$c" --timeout=60s >/dev/null || fail "CRD not Established: $c"
done
ok "GAIE CRDs present & Established"

info "Checking gateway controller in namespace '$GATEWAY_CONTROLLER_NAMESPACE' (${GATEWAY_LABEL_FILTER})…"
# namespace must exist
kubectl get ns "$GATEWAY_CONTROLLER_NAMESPACE" >/dev/null 2>&1 || fail "Namespace '$GATEWAY_CONTROLLER_NAMESPACE' not found (run step c Helm installs)"

PODS=$(kubectl get pods -n "$GATEWAY_CONTROLLER_NAMESPACE" -o name -l "$GATEWAY_LABEL_FILTER" 2>/dev/null)
# fallback label (charts sometimes label differently)
[[ -z "${PODS:-}" ]] && PODS=$(kubectl get pods -n "$GATEWAY_CONTROLLER_NAMESPACE" -o name | grep -E 'agentgateway|envoy-gateway|gateway' || true)
[[ -z "${PODS:-}" ]] && fail "gateway pods not found in '$GATEWAY_CONTROLLER_NAMESPACE'"

# pods should be running
for p in $PODS; do
  kubectl wait -n "$GATEWAY_CONTROLLER_NAMESPACE" --for=condition=Ready "$p" --timeout=180s >/dev/null || fail "Pod not Ready: $p"
done
ok "Gateway controller pods Ready ($PODS)"

kubectl get gateway.gateway.networking.k8s.io inference-gateway -n "$NAMESPACE" >/dev/null 2>&1 || fail "Gateway 'inference-gateway' not found in $NAMESPACE (apply step d manifest)"

ok "GAIE is installed and the gateway is up in namespace '$NAMESPACE'."
