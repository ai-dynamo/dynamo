#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/registry-mirror.yaml"
kubectl rollout status deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" --timeout=600s

if kubectl get pod glm52-eval-runner -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl exec glm52-eval-runner -n "${NAMESPACE}" -c runner -- \
    curl -fsS http://dockerhub-pull-cache:5000/v2/ >/dev/null
fi

echo "Docker Hub pull-through cache ready in ${NAMESPACE}"
