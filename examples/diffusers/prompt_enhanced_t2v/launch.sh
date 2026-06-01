#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm wrapper for the prompt-enhanced text-to-video example.
#
# Launches one enhancer LLM (Qwen3-0.6B by default) and N video generation
# replicas (Wan2.1-T2V-1.3B-Diffusers by default) on a single node. Each
# replica gets a distinct --master-port to avoid the sglang multimodal_gen
# 30005 default collision.

set -euo pipefail

PARTITION="${PARTITION:-gb200nvl72_cx8}"
WALLTIME="${WALLTIME:-04:00:00}"
IMAGE="${IMAGE:-nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.1.1}"
HOST_MOUNT="${HOST_MOUNT:-${HOME}}"
GUEST_MOUNT="${GUEST_MOUNT:-/scratch}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-${HOST_MOUNT}/exp-prompt-enhanced-t2v/logs}"
MEDIA_ROOT="${MEDIA_ROOT:-${HOST_MOUNT}/exp-prompt-enhanced-t2v/media}"
mkdir -p "${LOG_ROOT}" "${MEDIA_ROOT}"

sbatch --wait \
  --job-name=prompt-enhanced-t2v \
  --partition="${PARTITION}" \
  --nodes=1 \
  --gres=gpu:4 \
  --time="${WALLTIME}" \
  --output="${LOG_ROOT}/slurm-%j.out" \
  --container-image="${IMAGE}" \
  --container-mounts="${HOST_MOUNT}:${GUEST_MOUNT},${THIS_DIR}:/example" \
  --container-workdir=/example \
  --wrap 'bash /example/container.sh'
