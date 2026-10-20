# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a AS base

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    DEEPINFER_ACCELERATOR=nvidia \
    TORCH_CUDA_ARCH_LIST=10.0a \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang cmake curl git libclang-dev \
    libibverbs-dev libnuma-dev libssl-dev libzmq3-dev numactl pkg-config \
    protobuf-compiler python3-dev python3-venv rdma-core unzip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install 'torch==2.13.0+cu132' \
       --find-links https://download.pytorch.org/whl/cu132/torch/

FROM base AS deepinfer_builder
ARG DEEPINFER_REF=774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
ARG MAX_JOBS=4
ENV MAX_JOBS=$MAX_JOBS NVCC_THREADS=1
WORKDIR /opt/deepinfer
RUN git init . && git remote add origin https://github.com/deepseek-ai/DeepInfer.git \
    && git fetch --depth 1 origin "$DEEPINFER_REF" && git checkout --detach FETCH_HEAD \
    && git config --global url.https://github.com/.insteadOf git@github.com: \
    && git submodule update --init --recursive --depth 1 \
       vendor/DeepSelect vendor/FlashMLA vendor/DeepEP vendor/DeepGEMM

# Dynamo's frontend requires >=4.56; use the versions qualified by the local
# frontend smoke test, and record them in DeepInfer's built wheel metadata too.
RUN sed -i -e 's/^transformers==4.50.0$/transformers==4.57.6/' \
           -e 's/^tokenizers==0.21.4$/tokenizers==0.22.2/' requirements.txt \
    && python -m pip install -r requirements-nvidia.txt \
    && python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3 \
    && mkdir /wheels

# DeepSelect and FlashMLA's build-time test helpers call lspci even without GPUs.
RUN apt-get update && apt-get install -y --no-install-recommends pciutils \
    && rm -rf /var/lib/apt/lists/*
RUN cd vendor/DeepSelect && DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
    python setup.py bdist_wheel --dist-dir /wheels
# CUDA 13.2 spills in head-64 decode kernels; see the README qualification note.
RUN --mount=type=cache,target=/opt/deepinfer/vendor/FlashMLA/build \
    cd vendor/FlashMLA && FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA FLASH_MLA_SKIP_REG_SPILL_CHECK=1 \
    python setup.py bdist_wheel --dist-dir /wheels
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/opt/deepinfer/vendor/DeepEP/build \
    cd vendor/DeepEP && python setup.py bdist_wheel --dist-dir /wheels
RUN --mount=type=cache,target=/opt/deepinfer/vendor/DeepGEMM/build \
    cd vendor/DeepGEMM && DG_FORCE_BUILD=1 \
    python setup.py bdist_wheel --dist-dir /wheels
COPY examples/backends/deepinfer/small-prefill-candidates.patch /tmp/
RUN git apply /tmp/small-prefill-candidates.patch
RUN --mount=type=cache,target=/opt/deepinfer/build \
    python -m pip install --no-deps /wheels/*.whl \
    && python setup.py bdist_wheel --dist-dir /wheels

FROM base AS dynamo_builder
ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS \
    RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl --fail --show-error --silent https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.96.1 \
    && rm /tmp/rustup.sh \
    && python -m pip install 'maturin[patchelf]>=1.0,<2.0' build
WORKDIR /opt/dynamo
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .cargo/ .cargo/
COPY lib/ lib/
COPY examples/router/custom-policy-example/ examples/router/custom-policy-example/
COPY deploy/inference-gateway/ deploy/inference-gateway/
RUN apt-get update && apt-get install -y --no-install-recommends libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    --mount=type=cache,target=/opt/dynamo/lib/bindings/python/target \
    maturin build --release --strip --locked --interpreter "$VIRTUAL_ENV/bin/python" \
       --manifest-path lib/bindings/python/Cargo.toml --out /wheels --manylinux off
COPY pyproject.toml hatch_build.py README.md LICENSE ./
COPY components/src/dynamo/ components/src/dynamo/
RUN python -m build --wheel --outdir /wheels

FROM base AS runtime
ENV DYN_HTTP_HOST=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
# DeepInfer's TileLang operator modules import pytest during engine startup.
RUN --mount=type=bind,from=deepinfer_builder,source=/wheels,target=/tmp/deepinfer-wheels \
    --mount=type=bind,from=dynamo_builder,source=/wheels,target=/tmp/dynamo-wheels \
    python -m pip install /tmp/deepinfer-wheels/*.whl /tmp/dynamo-wheels/*.whl httpx nvidia-ml-py \
       flash-attn-4==4.0.0b34 fastapi==0.128.0 uvicorn==0.40.0 pytest==8.4.2 \
    # Upstream DeepInfer explicitly replaces Torch's NCCL 2.29.7 pin.
    && python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
WORKDIR /opt/dynamo
COPY examples/backends/deepinfer/ examples/backends/deepinfer/
COPY examples/common/launch_utils.sh examples/common/launch_utils.sh
ARG DEEPINFER_REF=774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
ARG DYNAMO_COMMIT=unknown
LABEL org.opencontainers.image.title="Dynamo with DeepInfer" \
      org.opencontainers.image.revision=$DYNAMO_COMMIT \
      io.deepinfer.revision=$DEEPINFER_REF
EXPOSE 8000
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "bash", "/opt/dynamo/examples/backends/deepinfer/launch/agg.sh"]
CMD []
