#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "usage: $0 start <suite> <remote-task-images.json> | status <suite>" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
mode=$1
suite=$2
[[ "${suite}" =~ ^[a-z0-9-]+$ ]] || usage
if [[ "${suite}" != verified ]]; then
  echo "This prefill implementation currently supports only SWE-bench Verified." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -d' ' -f1
  else
    shasum -a 256 | cut -d' ' -f1
  fi
}

current_cache_binding() {
  local marker_sha256 pvc_uid
  marker_sha256="$(kubectl exec deployment/dockerhub-pull-cache \
    -n "${NAMESPACE}" -c registry -- \
    sha256sum /artifacts/cache/.glm52-migration-v1.json | cut -d' ' -f1)"
  pvc_uid="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
    -o jsonpath='{.metadata.uid}')"
  printf '%s\n%s\n' "${marker_sha256}" "${pvc_uid}" | sha256_text
}

runner=glm52-eval-runner
session="glm52-cache-prefill-${suite}"
state_root="${GLM52_PREFILL_STATE_ROOT:-/workspace/glm52-cache-prefill}"
state_dir="${state_root}/${suite}"
status_path="${state_dir}/status.json"
remote_script="${state_dir}/prefill_swebench.py"
remote_lock_script="${state_dir}/prefill_lock.py"
log_path="${state_dir}/prefill.log"
lock_dir="/artifacts/glm52-nscale/.campaign-run.lock"
invocation_id="cache-prefill-${suite}-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
lock_acquired=0
handed_off=0

release_lock() {
  kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    python3 "${remote_lock_script}" release \
      --lock-dir "${lock_dir}" --invocation-id "${invocation_id}"
}

cleanup() {
  if ((lock_acquired == 1 && handed_off == 0)); then
    if ! release_lock; then
      echo "Failed to release prefill campaign lock ${invocation_id}" >&2
    fi
  fi
}
trap cleanup EXIT

show_status() {
  local active=false
  local binding
  binding="$(current_cache_binding)"
  if kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    tmux has-session -t "=${session}" 2>/dev/null; then
    active=true
  fi
  kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
    python3 - "${status_path}" "${active}" "${binding}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
active = sys.argv[2] == "true"
current_binding = sys.argv[3]
payload = json.loads(path.read_text()) if path.exists() else {"state": "not_started"}
payload["tmux_active"] = active
payload["current_cache_binding_sha256"] = current_binding
payload["cache_binding_matches"] = payload.get("cache_binding_sha256") == current_binding
print(json.dumps(payload, sort_keys=True))
PY
}

case "${mode}" in
  status)
    [[ $# -eq 2 ]] || usage
    "${SCRIPT_DIR}/assert-ready.sh" >/dev/null
    show_status
    exit 0
    ;;
  start)
    [[ $# -eq 3 ]] || usage
    manifest=$3
    ;;
  *)
    usage
    ;;
esac

"${SCRIPT_DIR}/assert-ready.sh" >/dev/null
if kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  tmux has-session -t "=${session}" 2>/dev/null; then
  owner="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    cat "${lock_dir}/owner.json" 2>/dev/null || true)"
  if ! jq -e --arg suite "${suite}" '
    .operation == "swebench-cache-prefill" and .suite == $suite
  ' <<<"${owner}" >/dev/null 2>&1; then
    echo "Prefill tmux session exists without its matching campaign lock." >&2
    exit 1
  fi
  echo "SWE-bench cache prefill is already active: ${session}"
  show_status
  exit 0
fi

"${ROOT_DIR}/eval/assert-runner-idle.sh"
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  test -s "${manifest}"
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p "${state_dir}"

temporary="${remote_script}.new"
kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
  sh -eu -c 'cat >"$1"; chmod 0555 "$1"; mv "$1" "$2"' \
  -- "${temporary}" "${remote_script}" <"${SCRIPT_DIR}/prefill_swebench.py"
local_sha256="$(sha256_file "${SCRIPT_DIR}/prefill_swebench.py")"
remote_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  sha256sum "${remote_script}" | cut -d' ' -f1)"
[[ "${local_sha256}" == "${remote_sha256}" ]]

lock_temporary="${remote_lock_script}.new"
kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
  sh -eu -c 'cat >"$1"; chmod 0555 "$1"; mv "$1" "$2"' \
  -- "${lock_temporary}" "${remote_lock_script}" <"${SCRIPT_DIR}/prefill_lock.py"
local_lock_sha256="$(sha256_file "${SCRIPT_DIR}/prefill_lock.py")"
remote_lock_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  sha256sum "${remote_lock_script}" | cut -d' ' -f1)"
[[ "${local_lock_sha256}" == "${remote_lock_sha256}" ]]

cache_binding_sha256="$(current_cache_binding)"

kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  python3 "${remote_lock_script}" acquire \
    --lock-dir "${lock_dir}" --invocation-id "${invocation_id}" \
    --suite "${suite}" --state-dir "${state_dir}"
lock_acquired=1

printf -v command \
  'set +e; python3 -u %q --manifest %q --state-dir %q --cache-binding-sha256 %q >>%q 2>&1; prefill_rc=$?; final_rc=$prefill_rc; if python3 %q record-exit --lock-dir %q --invocation-id %q --status %q --exit-code "$prefill_rc" >>%q 2>&1; then python3 %q release --lock-dir %q --invocation-id %q >>%q 2>&1 || final_rc=125; else final_rc=125; fi; exit "$final_rc"' \
  "${remote_script}" "${manifest}" "${state_dir}" "${cache_binding_sha256}" "${log_path}" \
  "${remote_lock_script}" "${lock_dir}" "${invocation_id}" "${status_path}" "${log_path}" \
  "${remote_lock_script}" "${lock_dir}" "${invocation_id}" "${log_path}"
# From this point onward a transport failure is ambiguous: the remote tmux may
# have started. Retain the lock unless the remote wrapper releases it.
handed_off=1
if ! kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  tmux new-session -d -s "${session}" "${command}"; then
  echo "Prefill launch transport failed; retaining the campaign lock for fail-closed recovery." >&2
  exit 1
fi
sleep 2
if ! kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  tmux has-session -t "=${session}" 2>/dev/null; then
  show_status
  kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    tail -n 50 "${log_path}" >&2 || true
  exit 1
fi

echo "SWE-bench cache prefill started: session=${session} script_sha256=${remote_sha256} lock_sha256=${remote_lock_sha256}"
show_status
