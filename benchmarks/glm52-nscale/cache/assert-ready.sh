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

runner=glm52-eval-runner
mirror=http://dockerhub-pull-cache:5000/

kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" -o json \
  | jq -e '.status.observedGeneration == .metadata.generation
    and .status.readyReplicas == 1
    and .status.availableReplicas == 1' >/dev/null
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  curl -fsS "${mirror}v2/" >/dev/null
mirrors="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker info --format '{{json .RegistryConfig.Mirrors}}')"
jq -e --arg mirror "${mirror}" 'index($mirror) != null' \
  <<<"${mirrors}" >/dev/null
cache_bytes="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  du -sb /artifacts/cache/dockerhub-registry | cut -f1)"

echo "Docker Hub pull-through cache PASS: mirror=${mirror} bytes=${cache_bytes}"
