#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve} {validation|ab|ba}" >&2
  exit 2
fi

variant="$1"
campaign_phase="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE NODE_NAME EXPECTED_GPU_UUIDS
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }
"${SCRIPT_DIR}/assert-pinned-source.sh" >/dev/null

case "${variant}" in
  dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve) ;;
  *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
esac
case "${campaign_phase}" in
  validation|ab|ba) ;;
  *) echo "unknown campaign phase: ${campaign_phase}" >&2; exit 2 ;;
esac

"${SCRIPT_DIR}/preflight.sh"
"${SCRIPT_DIR}/render.sh"
"${SCRIPT_DIR}/assert-clean-slate.sh"

if kubectl get pod -n "${NAMESPACE}" -o json \
  | jq -e '[.items[] | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber] | add > 0' \
  >/dev/null; then
  echo "Refusing to deploy while a GPU pod already exists in ${NAMESPACE}." >&2
  echo "Run ${SCRIPT_DIR}/teardown.sh first." >&2
  exit 1
fi

manifest="${SCRIPT_DIR}/rendered/${variant}.yaml"
kubectl apply -f "${manifest}"

case "${variant}" in
  dynamo-vllm|dynamo-sglang)
    name="glm52-${variant}"
    expected_generation="$(kubectl get dynamographdeployment "${name}" \
      -n "${NAMESPACE}" -o jsonpath='{.metadata.generation}')"
    kubectl wait --for=condition=Ready "dynamographdeployment/${name}" \
      -n "${NAMESPACE}" --timeout=3600s
    deadline=$((SECONDS + 3600))
    until kubectl get dynamographdeployment "${name}" -n "${NAMESPACE}" -o json \
      | jq -e --argjson generation "${expected_generation}" '
          .status.observedGeneration == $generation
          and .status.state == "successful"
          and ([.status.conditions[]?
            | select(.type == "Ready" and .status == "True")] | length == 1)
          and (.status.components | length == 2)
          and ([.status.components[]
            | select(.replicas == 1
              and .updatedReplicas == 1
              and .readyReplicas == 1)] | length == 2)
        ' >/dev/null; do
      if ((SECONDS >= deadline)); then
        echo "Timed out waiting for ${name} generation ${expected_generation}" >&2
        kubectl get dynamographdeployment "${name}" -n "${NAMESPACE}" -o yaml >&2
        exit 1
      fi
      sleep 5
    done
    ;;
  vllm-serve|sglang-serve)
    name="glm52-${variant}"
    kubectl rollout status "deployment/${name}" -n "${NAMESPACE}" --timeout=3600s
    ;;
esac

"${SCRIPT_DIR}/capture-identity.sh" "${variant}" "${campaign_phase}"
