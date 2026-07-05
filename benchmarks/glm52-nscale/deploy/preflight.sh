#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE NODE_NAME EXPECTED_GPU_UUIDS
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

kubectl get namespace "${NAMESPACE}" >/dev/null
kubectl get pvc shared-model-cache -n "${NAMESPACE}" -o jsonpath='{.status.phase}' | grep -qx Bound

if ! kubectl get secret nvcr-secret -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "Required image-pull secret ${NAMESPACE}/nvcr-secret is missing" >&2
  exit 1
fi

node_gpu_capacity="$(kubectl get node "${NODE_NAME}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
test "${node_gpu_capacity}" = "8"

requested_gpu="$(kubectl get pods -A -o json | jq --arg node "${NODE_NAME}" '
  [.items[]
   | select(.spec.nodeName == $node)
   | select(.status.phase == "Pending" or .status.phase == "Running")
   | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0"
   | tonumber] | add // 0
')"

free_gpu=$((node_gpu_capacity - requested_gpu))
if (( free_gpu < TP_SIZE )); then
  echo "Need ${TP_SIZE} scheduler-free GPUs on ${NODE_NAME}; found ${free_gpu}" >&2
  exit 1
fi

echo "Preflight PASS: namespace=${NAMESPACE} node=${NODE_NAME} scheduler_free_gpus=${free_gpu}"
echo "The worker's startup guard will independently reject physically dirty GPUs."
