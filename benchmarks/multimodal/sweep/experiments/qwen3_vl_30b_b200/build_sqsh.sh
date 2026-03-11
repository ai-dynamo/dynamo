#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build a .sqsh container image on a Slurm node via enroot import.
#
# Usage:
#   bash build_sqsh.sh --account <ACCOUNT>

set -euo pipefail

ACCOUNT=""
PARTITION="batch"
TIME="01:00:00"
LOCAL_IMAGE="/lustre/fsw/core_dlfw_ci/kprashanth/trtllm_bench_rc7_post1.sqsh"
IMAGE="gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:prashanth-trtllm-rc7-post1"

usage() {
    cat <<EOF
Usage: bash build_sqsh.sh --account ACCOUNT [OPTIONS]

Required:
  --account ACCOUNT       Slurm account

Options:
  --partition PARTITION   Slurm partition [default: $PARTITION]
  --time TIME             Slurm time limit [default: $TIME]
  --local-image PATH      Output .sqsh path [default: $LOCAL_IMAGE]
  --image IMAGE           Docker image to import [default: $IMAGE]
  --help                  Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)      ACCOUNT="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --local-image)  LOCAL_IMAGE="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown flag: $1"; usage ;;
    esac
done

if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: --account is required"
    exit 1
fi

echo "======================================================================="
echo "  Building .sqsh container image"
echo "======================================================================="
echo "  Image:       $IMAGE"
echo "  Output:      $LOCAL_IMAGE"
echo "  Account:     $ACCOUNT"
echo "  Partition:   $PARTITION"
echo "  Time:        $TIME"
echo ""

salloc \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --job-name="${ACCOUNT}-enroot-import" \
    -t "$TIME" \
    srun bash -l -c "ENROOT_TRANSFER_TIMEOUT=3600 enroot import -o '$LOCAL_IMAGE' 'docker://$IMAGE'"
