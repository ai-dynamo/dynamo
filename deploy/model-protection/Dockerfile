# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=protected-model-requires-base-image-arg
FROM ${BASE_IMAGE}
ARG VLLM_PROTECTED_VERSION=0.30.0
ARG DYNAMO_SOURCE_REVISION=unknown
ARG DYNAMO_SOURCE_DIRTY=unknown
LABEL org.opencontainers.image.revision=${DYNAMO_SOURCE_REVISION}
LABEL com.nvidia.dynamo.protected.vllm-version=${VLLM_PROTECTED_VERSION}
LABEL com.nvidia.dynamo.protected.source-dirty=${DYNAMO_SOURCE_DIRTY}

USER root
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 update \
    && DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        libtss2-esys-3.0.2-0t64 \
        libtss2-tcti-device0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY ai_dynamo_runtime-*.whl /tmp/
COPY check-runtime.py /tmp/check-runtime.py
RUN uv pip install --system --no-deps --reinstall /tmp/ai_dynamo_runtime-*.whl \
    && rm /tmp/ai_dynamo_runtime-*.whl \
    && EXPECTED_VLLM_VERSION="${VLLM_PROTECTED_VERSION}" python3 /tmp/check-runtime.py \
    && rm /tmp/check-runtime.py

USER 1000
