# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH
ARG SOURCE_ARCHIVE_SHA256
ARG DOCKERFILE_SHA256
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"
LABEL com.nvidia.dynamo.experiment="gms-cuda-init-uuid-isolation"
LABEL com.nvidia.dynamo.source-archive-sha256="${SOURCE_ARCHIVE_SHA256}"
LABEL com.nvidia.dynamo.dockerfile-sha256="${DOCKERFILE_SHA256}"

USER root
COPY lib/gpu_memory_service /tmp/gpu_memory_service
RUN UV_CACHE_DIR=/tmp/uv-cache SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    uv build --wheel --no-build-isolation \
        --out-dir /tmp/gms-wheel /tmp/gpu_memory_service \
    && uv pip install --system --reinstall --no-deps /tmp/gms-wheel/*.whl \
    && rm -rf /tmp/gpu_memory_service /tmp/gms-wheel /tmp/uv-cache
USER dynamo

ARG GMS_SERVER_GPU_UUID_ISOLATION=0
ENV DYN_GMS_SERVER_GPU_UUID_ISOLATION=${GMS_SERVER_GPU_UUID_ISOLATION}
ENV DYN_GMS_SERVER_EXPECTED_GPU_UUIDS=GPU-02ff0cc1-647f-dee7-8365-921738e945a6,GPU-0d5ad102-eb8f-922a-173e-e91033320e0f,GPU-9c595f65-4651-0b25-f95c-09a0abd5f5fa,GPU-4fba7684-5a96-6280-91ff-b41f7484564c,GPU-3ef7c092-d55c-ca6c-0018-9fc89ed28683,GPU-32a6c51c-d07d-7513-86b5-813b64e452d2,GPU-56c0be30-f1d3-a00d-8a2a-7fa70da8037f,GPU-ce231b92-c54f-3f0c-af1c-1a696db97f51
LABEL com.nvidia.dynamo.gms-server-gpu-uuid-isolation="${GMS_SERVER_GPU_UUID_ISOLATION}"
