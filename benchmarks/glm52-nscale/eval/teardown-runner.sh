#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE

if kubectl --context "${KUBE_CONTEXT}" get pod glm52-eval-runner \
  -n "${NAMESPACE}" >/dev/null 2>&1; then
  "${SCRIPT_DIR}/assert-runner-idle.sh"
fi
kubectl --context "${KUBE_CONTEXT}" delete pod glm52-eval-runner \
  -n "${NAMESPACE}" --ignore-not-found --wait=true
echo "Runner deleted; artifact PVC retained"
