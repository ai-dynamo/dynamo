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

POD="${EVAL_RUNNER_POD:-glm52-eval-runner}"
if ! kubectl get pod "${POD}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  exit 0
fi

active_processes="$(kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- python3 -c '
import os
from pathlib import Path

markers = (
    "/workspace/eval/terminalbench/",
    "/workspace/eval/swebench/",
    "/workspace/eval/bfcl/scripts/",
    "harbor run",
    "mini-swe-agent",
    "swebench.harness",
)
current = os.getpid()
for path in sorted(Path("/proc").glob("[0-9]*/cmdline")):
    pid = int(path.parent.name)
    if pid == current:
        continue
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except (OSError, ValueError):
        continue
    if command and any(marker in command for marker in markers):
        print(f"pid={pid} {command}")
')"
running_containers="$(kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  docker ps --format '{{.ID}} {{.Image}} {{.Names}}')"
tmux_sessions="$(kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  tmux list-sessions -F '#{session_name}' 2>/dev/null || true)"
lock_owner="$(kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -c 'if test -d /artifacts/glm52-nscale/.campaign-run.lock; then cat /artifacts/glm52-nscale/.campaign-run.lock/owner.json 2>/dev/null || echo "{\"present\":true}"; fi' \
  2>/dev/null || true)"

if [[ -n "${active_processes}" || -n "${running_containers}" \
  || -n "${tmux_sessions}" || -n "${lock_owner}" ]]; then
  echo "Evaluation runner is active; refusing to replace its pod or synced sources." >&2
  [[ -z "${active_processes}" ]] || printf '%s\n' "${active_processes}" >&2
  [[ -z "${running_containers}" ]] || printf 'running container: %s\n' "${running_containers}" >&2
  [[ -z "${tmux_sessions}" ]] || printf 'tmux session: %s\n' "${tmux_sessions}" >&2
  [[ -z "${lock_owner}" ]] || printf 'campaign lock: %s\n' "${lock_owner}" >&2
  exit 1
fi
