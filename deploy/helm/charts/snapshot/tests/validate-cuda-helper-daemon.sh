#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

disabled="$(helm template snapshot-daemon-disabled "${chart_dir}")"
if grep -q 'name: cuda-checkpoint-helper' <<<"${disabled}"; then
  echo "helper sidecar rendered while disabled" >&2
  exit 1
fi

for collision in \
  'storage.pvc.basePath=/' \
  'storage.pvc.basePath=/run' \
  'storage.pvc.basePath=/run/cuda-checkpoint-helper' \
  'storage.pvc.basePath=/run/cuda-checkpoint-helper/checkpoints' \
  'runtime.socketPath=/run/cuda-checkpoint-helper/runtime.sock' \
  'runtime.socketPath=/run/runtime.sock'; do
  if helm template invalid-collision "${chart_dir}" \
    --set config.cudaCheckpoint.storageMode=posix \
    --set config.cudaCheckpoint.daemon.enabled=true \
    --set-string "${collision}" >/dev/null 2>&1; then
    echo "daemon mode unexpectedly accepted overlapping path: ${collision}" >&2
    exit 1
  fi
done

helm template clean-paths "${chart_dir}" \
  --set config.cudaCheckpoint.storageMode=posix \
  --set config.cudaCheckpoint.daemon.enabled=true \
  --set-string storage.pvc.basePath=/checkpoints \
  --set-string runtime.socketPath=/var/run/containerd/containerd.sock >/dev/null

helm template pod-mount-no-agent-collision "${chart_dir}" \
  --set config.cudaCheckpoint.storageMode=posix \
  --set config.cudaCheckpoint.daemon.enabled=true \
  --set storage.accessMode=podMount \
  --set-string storage.pvc.basePath=/run/cuda-checkpoint-helper/checkpoints >/dev/null

for invalid_socket in \
  /helper.sock \
  /host/proc/helper.sock \
  /run/cuda-checkpoint-helper/../helper.sock \
  "/run/cuda-checkpoint-helper/$(printf 'x%.0s' {1..100}).sock"; do
  if helm template invalid-socket "${chart_dir}" \
    --set config.cudaCheckpoint.storageMode=posix \
    --set config.cudaCheckpoint.daemon.enabled=true \
    --set-string config.cudaCheckpoint.daemon.socketPath="${invalid_socket}" >/dev/null 2>&1; then
    echo "daemon mode unexpectedly accepted socket path: ${invalid_socket}" >&2
    exit 1
  fi
done

enabled="$(
  helm template snapshot-daemon-enabled "${chart_dir}" \
    --set config.cudaCheckpoint.storageMode=posix \
    --set config.cudaCheckpoint.daemon.enabled=true
)"
for expected in \
  'name: cuda-checkpoint-helper' \
  'emptyDir: {}' \
  '/run/cuda-checkpoint-helper/helper.sock' \
  'startupProbe:' \
  'readinessProbe:' \
  'livenessProbe:' \
  '"21600"' \
  'cpu: 250m' \
  'image: "nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.2.1"' \
  'daemonSocketPath: "/run/cuda-checkpoint-helper/helper.sock"'; do
  if ! grep -Fq "${expected}" <<<"${enabled}"; then
    echo "enabled helper render missing: ${expected}" >&2
    exit 1
  fi
done
if [[ "$(grep -c 'name: cuda-helper-socket' <<<"${enabled}")" -lt 3 ]]; then
  echo "shared socket volume is not mounted by both containers" >&2
  exit 1
fi

if helm template invalid-daemon "${chart_dir}" \
  --set config.cudaCheckpoint.daemon.enabled=true >/dev/null 2>&1; then
  echo "daemon mode unexpectedly accepted legacy CUDA storage" >&2
  exit 1
fi
