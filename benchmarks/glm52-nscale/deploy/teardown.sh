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

if kubectl get pod glm52-eval-runner -n "${NAMESPACE}" >/dev/null 2>&1; then
  "${ROOT_DIR}/eval/assert-runner-idle.sh"
fi

for name in glm52-dynamo-vllm glm52-dynamo-sglang; do
  kubectl delete dynamographdeployment "${name}" -n "${NAMESPACE}" \
    --ignore-not-found --wait=true
done

for name in glm52-vllm-serve glm52-sglang-serve; do
  kubectl delete deployment,service "${name}" -n "${NAMESPACE}" \
    --ignore-not-found --wait=true
done

deadline=$((SECONDS + 600))
tracked_resources=(
  pods services deployments dynamographdeployments.nvidia.com
  podcliques.grove.io podcliquesets.grove.io podgangs.scheduler.grove.io
)
remaining_resources() {
  local resource
  for resource in "${tracked_resources[@]}"; do
    if [[ "${resource}" == pods ]]; then
      kubectl get pods -n "${NAMESPACE}" -l 'glm52.nvidia.com/variant' \
        -o name 2>/dev/null || true
    else
      kubectl get "${resource}" -n "${NAMESPACE}" -o name 2>/dev/null \
        | grep -E '/glm52-' || true
    fi
  done
}

while [[ -n "$(remaining_resources)" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for GLM-5.2 resources to terminate in ${NAMESPACE}" >&2
    remaining_resources >&2
    exit 1
  fi
  sleep 5
done

gpu_request_total="$(kubectl get pods -n "${NAMESPACE}" -o json | jq '
  [.items[] | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0"
    | tonumber] | add // 0
')"
if ((gpu_request_total != 0)); then
  echo "Non-campaign GPU requests remain in ${NAMESPACE}: ${gpu_request_total}" >&2
  exit 1
fi

if kubectl get pod glm52-eval-runner -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl exec glm52-eval-runner -n "${NAMESPACE}" -c runner -- \
    rm -rf /artifacts/glm52-nscale/runtime-bindings
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
proof_dir="${ROOT_DIR}/results/runtime/teardown"
mkdir -p "${proof_dir}"
kubectl get pods -n "${NAMESPACE}" -o json \
  | jq --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --arg namespace "${NAMESPACE}" \
      '{captured_at: $captured_at,
        namespace: $namespace,
        gpu_request_total: ([.items[] | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add // 0),
        remaining_campaign_pods: [.items[]
          | select(.metadata.labels["glm52.nvidia.com/variant"] != null)
          | .metadata.name],
        remaining_campaign_resources: []}' \
    > "${proof_dir}/${timestamp}.json"

echo "Teardown PASS: no GLM-5.2 resources or GPU requests remain in ${NAMESPACE}"
