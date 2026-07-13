#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rendered="$(helm template snapshot "${chart_dir}")"

container_mounts_socket() {
  local wanted_container="$1"
  awk -v wanted_container="${wanted_container}" '
    /^      containers:$/ {
      in_containers = 1
      next
    }
    in_containers && /^      [^ ]/ {
      in_containers = 0
    }
    in_containers && /^        - name: / {
      container = $0
      sub(/^        - name: /, "", container)
      in_mounts = 0
      next
    }
    container == wanted_container && /^          volumeMounts:$/ {
      in_mounts = 1
      next
    }
    in_mounts && /^          [^ ]/ {
      in_mounts = 0
    }
    in_mounts && /^            - name: / {
      socket_mount = $0 == "            - name: cuda-helper-socket"
      next
    }
    in_mounts && socket_mount && /^              mountPath: "\/run\/cuda-checkpoint-helper"$/ {
      found_mount = 1
    }
    END {
      exit !found_mount
    }
  ' <<<"${rendered}"
}

for expected in \
  'name: cuda-checkpoint-helper' \
  '/usr/local/bin/cuda-checkpoint-helper' \
  '/run/cuda-checkpoint-helper/helper.sock' \
  'name: cuda-helper-socket' \
  'emptyDir: {}'; do
  if ! grep -qF "${expected}" <<<"${rendered}"; then
    echo "always-on CUDA helper rendering is missing: ${expected}" >&2
    exit 1
  fi
done

for container in agent cuda-checkpoint-helper; do
  if ! container_mounts_socket "${container}"; then
    echo "cuda-helper-socket is not mounted by ${container}" >&2
    exit 1
  fi
done

for removed in storageMode daemonSocketPath daemonFallback; do
  if grep -q "${removed}" <<<"${rendered}"; then
    echo "obsolete CUDA helper configuration was rendered: ${removed}" >&2
    exit 1
  fi
done

for invalid_path in \
  / \
  /run \
  /run/cuda-checkpoint-helper \
  /run/cuda-checkpoint-helper/checkpoints; do
  if helm template snapshot "${chart_dir}" \
    --set-string storage.pvc.basePath="${invalid_path}" >/dev/null 2>&1; then
    echo "agent-mounted storage unexpectedly overlaps helper path: ${invalid_path}" >&2
    exit 1
  fi
done
