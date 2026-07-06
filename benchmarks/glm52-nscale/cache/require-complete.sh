#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 || "$1" != verified ]]; then
  echo "usage: $0 verified <remote-task-images.json>" >&2
  exit 2
fi

suite=$1
manifest=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

runner=glm52-eval-runner
status="$("${SCRIPT_DIR}/prefill.sh" status "${suite}")"
manifest_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  sha256sum "${manifest}" | cut -d' ' -f1)"

if ! jq -e --arg suite "${suite}" --arg manifest_sha256 "${manifest_sha256}" '
  .state == "complete"
  and .suite == $suite
  and .total == 500
  and .completed == 500
  and .remaining == 0
  and .final_catalog_verified == 500
  and .tmux_active == false
  and .cache_binding_matches == true
  and .source_manifest_sha256 == $manifest_sha256
' <<<"${status}" >/dev/null; then
  echo "Verified cache prefill is incomplete or stale:" >&2
  jq . <<<"${status}" >&2
  exit 1
fi

echo "Verified cache prefill PASS: 500/500 manifest_sha256=${manifest_sha256}"
