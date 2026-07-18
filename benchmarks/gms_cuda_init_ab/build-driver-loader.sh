#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the committed Driver loader source}"
SOURCE_COMMIT=$(git -C "$ROOT" rev-parse "${SOURCE_COMMIT}^{commit}")
SOURCE_DATE_EPOCH=$(git -C "$ROOT" show -s --format=%ct "$SOURCE_COMMIT")
BASE_IMAGE=${BASE_IMAGE:-dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2}
IMAGE_REPOSITORY=${IMAGE_REPOSITORY:-dynamoci.azurecr.io/ai-dynamo/dynamo}
IMAGE_TAG=${IMAGE_TAG:-gms-vmm-host-arena-${SOURCE_COMMIT:0:12}}
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT

git -C "$ROOT" archive "$SOURCE_COMMIT" \
    lib/gpu_memory_service \
    benchmarks/gms_cuda_init_ab/Dockerfile \
    benchmarks/gms_cuda_init_ab/allocation-sizes.json \
    benchmarks/gms_cuda_init_ab/run-vmm-slab-probe.sh \
    benchmarks/gms_cuda_init_ab/vmm-slab-probe.py \
    | tar -x -C "$CONTEXT"
mv "$CONTEXT/benchmarks/gms_cuda_init_ab/Dockerfile" "$CONTEXT/Dockerfile"
SOURCE_ARCHIVE_SHA256=$(
    git -C "$ROOT" archive "$SOURCE_COMMIT" lib/gpu_memory_service \
        | sha256sum | cut -d' ' -f1
)
DOCKERFILE_SHA256=$(sha256sum "$CONTEXT/Dockerfile" | cut -d' ' -f1)
IMAGE="$IMAGE_REPOSITORY:$IMAGE_TAG"

docker build \
    --provenance=false \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" \
    --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
    --build-arg "SOURCE_ARCHIVE_SHA256=$SOURCE_ARCHIVE_SHA256" \
    --build-arg "DOCKERFILE_SHA256=$DOCKERFILE_SHA256" \
    --build-arg "GMS_SERVER_GPU_UUID_ISOLATION=1" \
    --tag "$IMAGE" \
    "$CONTEXT"

printf 'SOURCE_COMMIT=%s\nSOURCE_ARCHIVE_SHA256=%s\nDOCKERFILE_SHA256=%s\n' \
    "$SOURCE_COMMIT" \
    "$SOURCE_ARCHIVE_SHA256" \
    "$DOCKERFILE_SHA256"
printf 'EXPERIMENT_LOADER=%s\n' "$IMAGE"
