#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Inputs
NAMESPACE="${NAMESPACE:-default}"
HF_TOKEN="${HF_TOKEN:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
  cat << EOF
Usage:
  NAMESPACE=<ns> [HF_TOKEN=<token>] deploy/utils/setup_benchmarking_resources.sh

Sets up benchmarking and profiling resources in an existing Dynamo namespace:
  - Applies common manifests (ServiceAccount, Role, RoleBinding, PVC)
  - Creates HuggingFace token secret if HF_TOKEN provided
  - Installs benchmark dependencies if requirements.txt exists

Prerequisites:
  - Dynamo Cloud platform must already be installed in the namespace
  - kubectl must be configured and pointing to the target cluster

Environment variables:
  NAMESPACE         Target Kubernetes namespace (default: default)
  HF_TOKEN          Hugging Face token; if set, a secret named hf-token-secret is created (optional)
EOF
}

if ! command -v kubectl &>/dev/null; then err "kubectl not found"; exit 1; fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
  err "Namespace $NAMESPACE does not exist. Please create it first or install Dynamo Cloud platform."
  exit 1
fi

# Check if Dynamo platform is installed
if ! kubectl get pods -n "$NAMESPACE" | grep -q "dynamo-platform"; then
  warn "Dynamo platform pods not found in namespace $NAMESPACE"
  warn "Please ensure Dynamo Cloud platform is installed first:"
  warn "  See: docs/kubernetes/installation_guide.md"
  if [[ -z "${FORCE:-}" && -z "${YES:-}" ]]; then
    read -p "Continue anyway? [y/N]: " -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  else
    warn "Continuing due to FORCE/YES set."
  fi
fi

# Apply common manifests
log "Applying benchmarking manifests to namespace $NAMESPACE"
export NAMESPACE  # ensure envsubst can see it
for mf in "$(dirname "$0")/manifests"/*.yaml; do
  if [[ -f "$mf" ]]; then
    # Skip pvc-access-pod.yaml as it's managed by inject_manifest.py
    if [[ "$(basename "$mf")" == "pvc-access-pod.yaml" ]]; then
      log "Skipping $mf (managed by inject_manifest.py)"
      continue
    fi

    if command -v envsubst >/dev/null 2>&1; then
      envsubst < "$mf" | kubectl -n "$NAMESPACE" apply -f -
    else
      warn "envsubst not found; applying manifest without substitution: $mf"
      kubectl -n "$NAMESPACE" apply -f "$mf"
    fi
  fi
done
ok "Benchmarking manifests applied"

# Optional: Create Hugging Face token secret if HF_TOKEN provided
if [[ -n "$HF_TOKEN" ]]; then
  kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
  ok "hf-token-secret created/updated"
fi


ok "Benchmarking resource setup complete"

# Verify installation
log "Verifying installation..."
kubectl get serviceaccount dynamo-sa -n "$NAMESPACE" >/dev/null && ok "ServiceAccount dynamo-sa exists" || err "ServiceAccount dynamo-sa not found"
kubectl get pvc dynamo-pvc -n "$NAMESPACE" >/dev/null && ok "PVC dynamo-pvc exists" || err "PVC dynamo-pvc not found"

if [[ -n "$HF_TOKEN" ]]; then
  kubectl get secret hf-token-secret -n "$NAMESPACE" >/dev/null && ok "Secret hf-token-secret exists" || err "Secret hf-token-secret not found"
fi