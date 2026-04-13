# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG CUDA_RUNTIME_IMAGE=nvcr.io/nvidia/cuda:13.0.2-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG CUDA_PYTHON_VERSION=13.0.2

FROM python:${PYTHON_VERSION}-slim AS wheel-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY lib/gpu_memory_service/ /workspace/lib/gpu_memory_service/

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip wheel --no-deps \
        --wheel-dir /dist /workspace/lib/gpu_memory_service

FROM ${CUDA_RUNTIME_IMAGE}

ARG CUDA_PYTHON_VERSION

ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/gms

ENV PATH=/opt/gms/bin:${PATH}

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
        cuda-python==${CUDA_PYTHON_VERSION} \
        nvidia-ml-py

COPY --from=wheel-builder /dist/gpu_memory_service*.whl /tmp/

RUN python3 -m pip install /tmp/gpu_memory_service*.whl \
    && rm -f /tmp/gpu_memory_service*.whl

ENTRYPOINT ["gpu-memory-service"]
