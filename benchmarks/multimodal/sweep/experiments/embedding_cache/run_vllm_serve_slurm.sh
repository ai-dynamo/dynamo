#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

IMAGE="nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.0.0rc8-arm64"
WORKSPACE="/home/scratch.qiwa_ent/workspace/dynamo"
CONFIG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 --config <yaml>"
  exit 1
fi

ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"

PATCH_SCRIPT=$(cat <<'PATCH'
cd /opt/dynamo/venv/lib/python3.12/site-packages

echo "Applying vllm PR #34182..."
curl -sL https://github.com/vllm-project/vllm/pull/34182.diff | patch -p1 || true

echo "Applying vllm PR #34783 (vllm/ only)..."
curl -sL https://github.com/vllm-project/vllm/pull/34783.diff | python3 -c "
import sys
chunks = sys.stdin.read().split('diff --git ')
filtered = [c for c in chunks if c.startswith('a/vllm/')]
print(''.join('diff --git ' + c for c in filtered))
" | patch -p1 || true
PATCH
)

CONTAINER_MOUNTS="\
/home/scratch.qiwa_ent/:/home/qiwa,\
/home/scratch.qiwa_ent/workspace/dynamo:/workspace,\
/home/scratch.qiwa_ent/huggingface:/huggingface"

srun \
  --overlap \
  --container-image "$IMAGE" \
  --container-mounts "$CONTAINER_MOUNTS" \
  --jobid "$SLURM_JOB_ID" \
  -A "$ACCOUNT" \
  bash -c "$PATCH_SCRIPT && cd /workspace && python -m benchmarks.multimodal.sweep --config $CONFIG"
