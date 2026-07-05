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

campaign_name_pattern='/glm52-(dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve)($|-)'
resources=(
  pods services deployments replicasets statefulsets
  dynamographdeployments.nvidia.com
  podcliques.grove.io podcliquesets.grove.io podgangs.scheduler.grove.io
)
existing=()
for resource in "${resources[@]}"; do
  output="$(kubectl get "${resource}" -n "${NAMESPACE}" -o name)"
  while IFS= read -r name; do
    [[ -n "${name}" ]] || continue
    if [[ "${name}" =~ ${campaign_name_pattern} ]]; then
      existing+=("${name}")
    fi
  done <<< "${output}"
done

if ((${#existing[@]} != 0)); then
  echo "Refusing to reuse existing GLM-5.2 serving resources; run teardown.sh first:" >&2
  printf '  %s\n' "${existing[@]}" >&2
  exit 1
fi

echo "Clean-slate PASS: no campaign serving resources exist in ${NAMESPACE}"
