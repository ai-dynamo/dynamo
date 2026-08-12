#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -eq 0 ]; then
  echo "usage: snapshot-cuda-vmm-launch COMMAND [ARG...]" >&2
  exit 2
fi

shim=/usr/local/lib/snapshot/libsnapshot_cuda_vmm.so
if [ ! -r "$shim" ]; then
  echo "CUDA VMM snapshot shim not found: $shim" >&2
  exit 1
fi
if [ -z "${DYN_SNAPSHOT_JOB_ID:-}" ]; then
  echo "DYN_SNAPSHOT_JOB_ID is required" >&2
  exit 2
fi

export DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1
export LD_PRELOAD="$shim${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$@"
