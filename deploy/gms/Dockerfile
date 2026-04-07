# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG CUDA_RUNTIME_IMAGE=nvcr.io/nvidia/cuda:13.0.2-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG CUDA_PYTHON_VERSION=13.0.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

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

ARG PYTHON_VERSION
ARG CUDA_PYTHON_VERSION
ARG TORCH_INDEX_URL

ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

RUN python3 -m venv /opt/gms

ENV PATH=/opt/gms/bin:${PATH}

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
        --extra-index-url ${TORCH_INDEX_URL} \
        cuda-python==${CUDA_PYTHON_VERSION} \
        msgspec \
        numpy \
        nvidia-ml-py \
        uvloop \
        torch

COPY --from=wheel-builder /dist/gpu_memory_service*.whl /tmp/

RUN python3 -m pip install --no-deps /tmp/gpu_memory_service*.whl \
    && rm -f /tmp/gpu_memory_service*.whl

ENTRYPOINT ["python3", "-m", "gpu_memory_service"]
