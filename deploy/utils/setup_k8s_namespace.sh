#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Inputs
NAMESPACE="${NAMESPACE:-default}"
DOCKER_SERVER="${DOCKER_SERVER:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_PASSWORD="${DOCKER_PASSWORD:-}"
HF_TOKEN="${HF_TOKEN:-}"
PULL_SECRET_NAME=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }

create_or_update_pull_secret() {
  local server="$1"; local user="$2"; local pass="$3"
  if [[ -n "$server" && -n "$user" && -n "$pass" ]]; then
    log "Creating/updating docker-imagepullsecret for $server in namespace $NAMESPACE"
    kubectl create secret docker-registry docker-imagepullsecret \
      --docker-server="$server" \
      --docker-username="$user" \
      --docker-password="$pass" \
      --namespace="$NAMESPACE" \
      --dry-run=client -o yaml | kubectl apply -f -
    ok "docker-imagepullsecret configured"
    PULL_SECRET_NAME="docker-imagepullsecret"
  fi
}

usage() {
  cat << EOF
Usage:
  NAMESPACE=<ns> deploy/utils/setup_k8s_namespace.sh
  NAMESPACE=<ns> DOCKER_SERVER=<registry> IMAGE_TAG=<tag> [DOCKER_USERNAME=<user>] [DOCKER_PASSWORD=<token>] \
    deploy/utils/setup_k8s_namespace.sh

Sets up Kubernetes namespace for Dynamo (one-time per namespace):
  - Creates namespace if absent
  - Applies common manifests (ServiceAccount, Role, RoleBinding, PVC)
  - Installs CRDs once per cluster (if not already installed)
  - If DOCKER_SERVER/IMAGE_TAG are provided:
      * Builds/pushes a custom operator image with Earthly
      * Installs/updates the operator Helm release using that image
      * If credentials (DOCKER_USERNAME/DOCKER_PASSWORD) are provided, creates/updates docker-imagepullsecret
      * If credentials are not provided, prompts interactively to create the pull secret
  - Otherwise installs the operator using default image: nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.1

Environment variables:
  NAMESPACE         Target Kubernetes namespace (default: default)
  DOCKER_SERVER     Registry server for operator image (optional)
  IMAGE_TAG         Image tag for operator (optional)
  DOCKER_USERNAME   Registry username (optional; if provided with DOCKER_PASSWORD, secret is created)
  DOCKER_PASSWORD   Registry password/token (optional)
  HF_TOKEN          Hugging Face token; if set, a secret named hf-token-secret is created in the namespace (optional)
EOF
}

if ! command -v kubectl &>/dev/null; then err "kubectl not found"; exit 1; fi

# 1) Ensure namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
  log "Creating namespace $NAMESPACE"
  kubectl create namespace "$NAMESPACE"
else
  log "Namespace $NAMESPACE exists"
fi

# 2) Apply common manifests
log "Applying common manifests to namespace $NAMESPACE"
for mf in "$(dirname "$0")/manifests"/*.yaml; do
  envsubst < "$mf" | kubectl apply -f -
done
ok "Common manifests applied"

# 3) Install CRDs once per cluster (only if not already installed)
if command -v helm &>/dev/null; then
  if ! helm status dynamo-crds -n "$NAMESPACE" &>/dev/null; then
    log "Installing CRDs via Helm release dynamo-crds in namespace $NAMESPACE"
    pushd "$REPO_ROOT/deploy/cloud/helm" >/dev/null
    helm upgrade --install dynamo-crds ./crds/ \
      --namespace "$NAMESPACE" \
      --wait \
      --atomic
    popd >/dev/null
    ok "CRDs installed"
  fi
fi

# 4) Optional: Create Hugging Face token secret if HF_TOKEN provided
if [[ -n "$HF_TOKEN" ]]; then
  kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
  ok "hf-token-secret created/updated"
fi

# 5) Optional: Create imagePullSecret for private registry if credentials provided or requested
if [[ -n "$DOCKER_SERVER" ]]; then
  if [[ -n "$DOCKER_USERNAME" && -n "$DOCKER_PASSWORD" ]]; then
    create_or_update_pull_secret "$DOCKER_SERVER" "$DOCKER_USERNAME" "$DOCKER_PASSWORD"
  elif [[ -n "$IMAGE_TAG" ]]; then
    echo
    read -p "Do you need image pull credentials for $DOCKER_SERVER (private registry)? [y/N]: " -r ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      read -p "Docker username (often '$oauthtoken' for NGC): " DOCKER_USERNAME
      read -s -p "Docker password/token: " DOCKER_PASSWORD; echo
      if [[ -n "$DOCKER_USERNAME" && -n "$DOCKER_PASSWORD" ]]; then
        create_or_update_pull_secret "$DOCKER_SERVER" "$DOCKER_USERNAME" "$DOCKER_PASSWORD"
      else
        warn "Username or password empty; skipping secret creation"
      fi
    fi
  fi
fi

# 6) Operator: Build/push custom image if both vars provided, else use default NGC image
if [[ -n "$DOCKER_SERVER" && -n "$IMAGE_TAG" ]]; then
  if ! command -v earthly &>/dev/null; then warn "earthly not found; skipping operator build/push"; else
    log "Building and pushing operator images via earthly"
    earthly --push +all-docker --DOCKER_SERVER="$DOCKER_SERVER" --IMAGE_TAG="$IMAGE_TAG"
  fi

  if ! command -v helm &>/dev/null; then warn "helm not found; skipping helm install"; else
    pushd "$REPO_ROOT/deploy/cloud/helm/platform" >/dev/null
    helm dep build
    popd >/dev/null

    pushd "$REPO_ROOT/deploy/cloud/helm" >/dev/null
    # Build Helm args
    HELM_ARGS=(upgrade dynamo-platform ./platform/ --install --namespace "$NAMESPACE" \
      --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
      --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}")
    if [[ -n "$PULL_SECRET_NAME" ]]; then
      HELM_ARGS+=(--set "dynamo-operator.imagePullSecrets[0].name=${PULL_SECRET_NAME}")
    fi
    helm "${HELM_ARGS[@]}"
    popd >/dev/null
    ok "Helm chart installed/updated"
  fi
else
  # Use default published image when custom not provided
  DEFAULT_OPERATOR_IMAGE="nvcr.io/nvidia/ai-dynamo/kubernetes-operator:0.4.1"
  if ! command -v helm &>/dev/null; then warn "helm not found; skipping helm install"; else
    pushd "$REPO_ROOT/deploy/cloud/helm/platform" >/dev/null
    helm dep build
    popd >/dev/null

    pushd "$REPO_ROOT/deploy/cloud/helm" >/dev/null
    # Only set imagePullSecrets if the referenced secret exists; otherwise rely on SA
    HELM_ARGS=(upgrade dynamo-platform ./platform/ --install --namespace "$NAMESPACE" \
      --set "dynamo-operator.controllerManager.manager.image.repository=${DEFAULT_OPERATOR_IMAGE%:*}" \
      --set "dynamo-operator.controllerManager.manager.image.tag=${DEFAULT_OPERATOR_IMAGE##*:}")
    if kubectl get secret nvcr-imagepullsecret -n "$NAMESPACE" &>/dev/null; then
      HELM_ARGS+=(--set "dynamo-operator.imagePullSecrets[0].name=nvcr-imagepullsecret")
    fi
    helm "${HELM_ARGS[@]}"
    popd >/dev/null
    ok "Helm chart installed/updated with default operator image"
  fi
fi

# 7) Install benchmark dependencies if requirements.txt exists
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [[ -f "$REQUIREMENTS_FILE" ]]; then
  log "Installing benchmark dependencies..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install -r "$REQUIREMENTS_FILE"
  elif command -v pip3 >/dev/null 2>&1; then
    pip3 install -r "$REQUIREMENTS_FILE"
  elif command -v pip >/dev/null 2>&1; then
    pip install -r "$REQUIREMENTS_FILE"
  else
    warn "No pip/pip3/uv found; skipping benchmark dependency installation"
    warn "To run benchmarks, manually install: pip install -r $REQUIREMENTS_FILE"
  fi
  ok "Benchmark dependencies installed"
fi

ok "Kubernetes namespace setup complete"
