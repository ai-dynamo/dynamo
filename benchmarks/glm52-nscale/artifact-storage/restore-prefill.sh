#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 LOCAL_VERIFIED_CHECKPOINT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

source_dir="$(cd "$1" && pwd)"
runner=glm52-eval-runner
remote_parent=/workspace/glm52-cache-prefill
remote_dir=${remote_parent}/verified
manifest=/artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified/task-images.json

"${ROOT_DIR}/eval/assert-runner-idle.sh"
"${SCRIPT_DIR}/assert-ready.sh" >/dev/null
local_identity="$(python3 "${SCRIPT_DIR}/checkpoint_identity.py" "${source_dir}")"
jq -e '
  .files == 507
  and .directories == 1
  and .bytes == 804078
  and .tree_sha256 == "b529bad556711d1b0f65de264dad25bca1c02d370365445791400bb9bb4d18ec"
' <<<"${local_identity}" >/dev/null
jq -e '
  .state == "complete"
  and .suite == "verified"
  and .total == 500
  and .completed == 500
  and .remaining == 0
  and .final_catalog_verified == 500
  and .cache_binding_sha256 == "7173f3ba3fdc1f94a41ce1736ca064729ca1274b8de13faa1137655a65a66ad8"
' "${source_dir}/status.json" >/dev/null

kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  rm -rf "${remote_dir}"
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p "${remote_parent}"
kubectl cp -c runner "${source_dir}" \
  "${NAMESPACE}/${runner}:${remote_dir}"
remote_identity="$(kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
  python3 - "${remote_dir}" <"${SCRIPT_DIR}/checkpoint_identity.py")"
if [[ "${remote_identity}" != "${local_identity}" ]]; then
  echo "Restored prefill checkpoint identity differs: local=${local_identity} remote=${remote_identity}" >&2
  exit 1
fi
"${ROOT_DIR}/cache/require-complete.sh" verified "${manifest}"
echo "Prefill checkpoint restore PASS: ${remote_identity}"
