#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# NUMA/GPU helpers for the qwen3-tp2-experiments bundle. Pure functions — no
# side effects at source time.
#
# Topology this bundle assumes (4xGB200, two NUMA nodes):
#   GPUs 0,1 -> NUMA 0
#   GPUs 2,3 -> NUMA 1
#
# Verified on the target host via `nvidia-smi topo -m` (NUMA Affinity column).
# /sys/bus/pci/devices/*/numa_node returns no-such-file on this kernel; we lean
# on nvidia-smi as the source of truth for the runtime check.

# Map a single GPU index -> NUMA node. Falls back to the prescribed map if
# nvidia-smi can't be probed.
gpu_numa_node() {
  case "$1" in 0|1) echo 0;; 2|3) echo 1;; *) echo 0;; esac
}

# Probe the actual GPU->NUMA map via `nvidia-smi topo -m` and bail if it
# disagrees with the prescribed (0,1)->0 / (2,3)->1 layout. Prints the parsed
# topology so a mismatch is easy to diagnose.
verify_numa_topology() {
  local out
  out=$(nvidia-smi topo -m 2>/dev/null) || {
    echo "[numa-lib] nvidia-smi topo -m failed — cannot verify topology" >&2
    return 1
  }
  # Parse rows like:
  #   GPU0   X  NV18 NV18 NV18 NODE NODE SYS SYS  0-69    0      N/A
  # The 'NUMA Affinity' column is the last numeric one before "N/A".
  local mismatch=0
  for gpu in 0 1 2 3; do
    local expected; expected=$(gpu_numa_node "$gpu")
    local actual
    actual=$(echo "$out" | awk -v g="GPU$gpu" '
      $1 == g {
        for (i = NF; i > 0; i--) {
          if ($i ~ /^[0-9]+$/) { print $i; exit }
        }
      }')
    if [ -z "$actual" ]; then
      echo "[numa-lib] could not parse NUMA Affinity for GPU$gpu from nvidia-smi" >&2
      mismatch=1
      continue
    fi
    if [ "$actual" != "$expected" ]; then
      echo "[numa-lib] GPU$gpu: expected NUMA $expected, nvidia-smi says NUMA $actual" >&2
      mismatch=1
    fi
  done
  if [ "$mismatch" -ne 0 ]; then
    echo "[numa-lib] FAILING: GPU NUMA topology does not match (0,1)->0 (2,3)->1" >&2
    echo "$out" | grep -E '^(GPU|.+CPU Affinity)' >&2
    return 1
  fi
  return 0
}

