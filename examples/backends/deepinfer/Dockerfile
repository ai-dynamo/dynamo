# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DeepInfer (github.com/deepseek-ai/DeepInfer) + Dynamo worker image for Blackwell (sm_100a).
# DeepInfer's kernels are sm_100a/sm_103a only. Build natively on the target arch (GB200 and
# GB300 are linux/arm64); FlashMLA needs a lot of RAM, so size MAX_JOBS/NVCC_THREADS to the
# build host.
# Build context is the Dynamo repo root:
#   docker buildx build --platform linux/arm64 -f examples/backends/deepinfer/Dockerfile -t <image> .
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04

ARG DEEPINFER_REF=774ce24
# Dynamo wheels for everything but dynamo.deepinfer, which comes from this checkout below.
ARG DYNAMO_VERSION=1.6.0.dev20261019
ARG MAX_JOBS=32
ARG NVCC_THREADS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    DEEPINFER_ACCELERATOR=nvidia \
    TORCH_CUDA_ARCH_LIST=10.0a

# pciutils: DeepSelect's setup.py runs lspci. libdw-dev: DeepEP/DeepGEMM JIT. libnuma1: NUMA binding.
# Verbs userspace for NCCL GIN and UCX over RDMA NICs (rdma-core itself pulls in systemd);
# infiniband-diags: DeepEP reads RDMA bandwidth from ibstat (0 without it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libdw-dev libnuma1 pciutils \
        python3.12 python3.12-dev python3.12-venv \
        libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 infiniband-diags \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# DeepInfer at a pinned ref. FlashMLA's register-spill guard trips under CUDA 13.2+.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/deepseek-ai/DeepInfer.git /opt/DeepInfer \
    && cd /opt/DeepInfer \
    && git checkout ${DEEPINFER_REF} \
    && pip install setuptools wheel \
    && MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} FLASH_MLA_SKIP_REG_SPILL_CHECK=1 ./build.sh

# Dynamo worker: the nightly wheels carry the Rust unified-backend Worker. --no-deps keeps
# DeepInfer's pins (transformers/tokenizers); the in-process worker path does not import them.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "ai-dynamo-runtime==${DYNAMO_VERSION}" "ai-dynamo==${DYNAMO_VERSION}" \
    && pip install "pydantic==2.13.5" "uvloop>=0.21.0" "msgspec>=0.19.0" \
        "prometheus-client>=0.23.1,<1"

# Frontend in its own venv, with ai-dynamo's full dependency set.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m venv /opt/dynamo-frontend \
    && /opt/dynamo-frontend/bin/pip install --extra-index-url https://pypi.nvidia.com \
        "ai-dynamo-runtime==${DYNAMO_VERSION}" "ai-dynamo==${DYNAMO_VERSION}"

# This checkout's dynamo.deepinfer. `dynamo` is a namespace package, so the overlay merges with
# the installed wheel; mount a newer copy over /opt/dynamo-deepinfer to iterate without
# rebuilding.
COPY components/src/dynamo/deepinfer /opt/dynamo-deepinfer/dynamo/deepinfer

ENV PYTHONPATH=/opt/dynamo-deepinfer:/opt/DeepInfer
WORKDIR /opt/DeepInfer
