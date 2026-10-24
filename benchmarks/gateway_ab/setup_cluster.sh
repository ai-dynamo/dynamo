#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the cluster-wide prerequisites for the Frontend vs Gateway+EPP A/B
# benchmark: the Dynamo platform (operator + CRDs) and the Gateway API
# Inference Extension stack (Gateway API CRDs, GAIE CRDs, agentgateway, and a
# Gateway named `inference-gateway`).
#
# Idempotent: re-running upgrades in place. Every kubectl/helm call pins
# --kube-context when KUBE_CONTEXT is set, so a concurrent context switch in
# another shell cannot redirect the install.
#
# Usage:
#   KUBE_CONTEXT=my-cluster bash benchmarks/gateway_ab/setup_cluster.sh
#
# Environment:
#   KUBE_CONTEXT        kubectl context to install into (default: current context)
#   DYNAMO_VERSION      Dynamo platform chart version (default: 1.5.1)
#   PLATFORM_NAMESPACE  namespace for the Dynamo operator (default: dynamo-system)
#   BENCH_NAMESPACE     workload namespace to create (default: gateway-ab)
#   AGW_NAMESPACE       namespace for agentgateway and the Gateway (default: agentgateway-system)
#   SKIP_PLATFORM=1     skip the Dynamo platform install (an operator already runs)
#   SKIP_GATEWAY=1      skip the Gateway API / GAIE / agentgateway install

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DYNAMO_VERSION=${DYNAMO_VERSION:-1.5.1}
PLATFORM_NAMESPACE=${PLATFORM_NAMESPACE:-dynamo-system}
BENCH_NAMESPACE=${BENCH_NAMESPACE:-gateway-ab}
AGW_NAMESPACE=${AGW_NAMESPACE:-agentgateway-system}

for tool in kubectl helm; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found on PATH" >&2; exit 1; }
done

if [[ -n "${KUBE_CONTEXT:-}" ]]; then
  KUBECTL=(kubectl --context "$KUBE_CONTEXT")
  HELM=(helm --kube-context "$KUBE_CONTEXT")
else
  KUBE_CONTEXT=$(kubectl config current-context)
  KUBECTL=(kubectl --context "$KUBE_CONTEXT")
  HELM=(helm --kube-context "$KUBE_CONTEXT")
fi
echo "Installing into context: $KUBE_CONTEXT"

"${KUBECTL[@]}" create namespace "$BENCH_NAMESPACE" --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f -

if [[ "${SKIP_PLATFORM:-0}" != "1" ]]; then
  if "${KUBECTL[@]}" get deployment -A -l app.kubernetes.io/name=dynamo-operator -o name 2>/dev/null \
      | grep -q . && ! "${HELM[@]}" status dynamo-platform -n "$PLATFORM_NAMESPACE" >/dev/null 2>&1; then
    echo "error: a Dynamo operator already runs in this cluster under another release." >&2
    echo "       Re-run with SKIP_PLATFORM=1 to use it, or remove it first." >&2
    exit 1
  fi
  echo "Installing dynamo-platform $DYNAMO_VERSION into $PLATFORM_NAMESPACE"
  "${HELM[@]}" upgrade --install dynamo-platform \
    "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${DYNAMO_VERSION}.tgz" \
    --namespace "$PLATFORM_NAMESPACE" --create-namespace \
    --wait --timeout 10m
fi

if [[ "${SKIP_GATEWAY:-0}" != "1" ]]; then
  # The repository installer uses the current context, so run it with a
  # kubeconfig whose current context is pinned to ours. --flatten embeds the
  # certificates, whose paths may be relative to the original kubeconfig.
  pinned_kubeconfig=$(mktemp)
  trap 'rm -f "$pinned_kubeconfig"' EXIT
  kubectl config view --raw --minify --flatten --context "$KUBE_CONTEXT" > "$pinned_kubeconfig"
  KUBECONFIG="$pinned_kubeconfig" NAMESPACE="$BENCH_NAMESPACE" AGW_NAMESPACE="$AGW_NAMESPACE" \
    bash "$REPO_ROOT/deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh"
fi

echo
echo "Prerequisites installed. Next:"
echo "  python3 -m benchmarks.gateway_ab --context $KUBE_CONTEXT --namespace $BENCH_NAMESPACE \\"
echo "    --gateway-namespace $AGW_NAMESPACE preflight"
echo "  (run from the repository root)"
