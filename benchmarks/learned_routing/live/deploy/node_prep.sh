#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-node preparation on the allocated host (one task per node): node-local root, hardware
# record and checks, and a node-local copy of the vLLM image for Pyxis.
# Usage: node_prep.sh RUN_DIR
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 1 ]] || die "usage: node_prep.sh RUN_DIR"
host="$(hostname)"
out="$1/nodes/$host"
mkdir -p "$out"
if [[ ! -w "$LR_NODE_BASE" ]]; then
  die "$LR_NODE_BASE is not writable on $host; set LR_NODE_BASE (e.g. /tmp, which is RAM)"
fi
mkdir -p "$LR_NODE_ROOT"
{
  date -u +%FT%TZ
  uname -a
  grep PRETTY_NAME /etc/os-release
  ldd --version | head -1
  nproc
  free -g | head -2
  df -h "$LR_NODE_ROOT" "$LR_SHARED_ROOT" | tail -2
} > "$out/host.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader > "$out/gpus.csv"
nvidia-smi topo -m > "$out/topo.txt" 2>&1 || true
lscpu > "$out/lscpu.txt" 2>&1 || true
gpus=$(wc -l < "$out/gpus.csv")
h100=$(grep -c 'H100 80GB HBM3' "$out/gpus.csv" || true)
[[ "$gpus" -eq 8 && "$h100" -eq 8 ]] || die "$host: expected 8 x H100 80GB HBM3, got $gpus GPUs ($h100 H100 SXM)"
if [[ "$(stat -c %s "$LR_NODE_SQSH" 2>/dev/null || echo 0)" != "$LR_IMAGE_SQSH_BYTES" ]]; then
  started=$(date +%s)
  cp "$LR_IMAGE_SQSH" "$LR_NODE_SQSH.partial"
  mv "$LR_NODE_SQSH.partial" "$LR_NODE_SQSH"
  echo "image copy seconds=$(($(date +%s) - started))" >> "$out/host.txt"
fi
[[ "$(stat -c %s "$LR_NODE_SQSH")" == "$LR_IMAGE_SQSH_BYTES" ]] || die "$host: image copy has the wrong size"
echo "node_prep ok host=$host node_root=$LR_NODE_ROOT"
