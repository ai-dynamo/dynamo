#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE

if [[ $# -eq 0 ]]; then
  set -- bash
fi

terminal_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  terminal_args+=(-t)
fi
exec kubectl --context "${KUBE_CONTEXT}" exec "${terminal_args[@]}" -n "${NAMESPACE}" \
  glm52-eval-runner -c runner -- "$@"
