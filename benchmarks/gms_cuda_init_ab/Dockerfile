# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
FROM ${BASE_IMAGE}

ARG SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"
LABEL com.nvidia.dynamo.experiment="gms-cuda-init-uuid-isolation"

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
LABEL com.nvidia.dynamo.gms-server-gpu-uuid-isolation="${GMS_SERVER_GPU_UUID_ISOLATION}"
