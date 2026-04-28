# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-stage build: stage 1 compiles the dynamo._core Rust wheel; stage 2
# layers the wheel + dynamo Python package onto the upstream vLLM image.
# Dynamo source is cloned from github at build time, so the build context is
# small and the image is reproducible regardless of local working-tree state.
#
# Build:
#   docker build -t <registry>/nemotron-omni-vllm:latest \
#                -f recipes/nemotron-omni/Dockerfile recipes/nemotron-omni
#
# Override defaults with --build-arg, e.g.:
#   --build-arg BASE_IMAGE=vllm/vllm-openai:v0.20.0
#   --build-arg DYNAMO_COMMIT=<sha-or-branch>

ARG BASE_IMAGE="vllm/vllm-openai:v0.20.0"
ARG DYNAMO_COMMIT="main"
ARG CARGO_BUILD_JOBS="16"

# ======== Stage 1: Builder (Rust toolchain + maturin wheel) ========
FROM ${BASE_IMAGE} AS builder
ARG DYNAMO_COMMIT
ARG CARGO_BUILD_JOBS
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}

USER root
WORKDIR /build

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential pkg-config libclang-dev protobuf-compiler git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.93.1

ENV CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH=/root/.cargo/bin:${PATH}

RUN git clone https://github.com/ai-dynamo/dynamo.git /build/dynamo && \
    cd /build/dynamo && git checkout ${DYNAMO_COMMIT}

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    pip install --no-cache-dir maturin && \
    cd /build/dynamo/lib/bindings/python && \
    maturin build --release && \
    mkdir -p /build/dist && \
    cp target/wheels/*.whl /build/dist/

# ======== Stage 2: Final image ========
FROM ${BASE_IMAGE}
USER root

COPY --from=builder /build/dynamo /workspace/dynamo
COPY --from=builder /build/dist/*.whl /tmp/

# Install the dynamo._core wheel, then the dynamo Python package in editable
# mode (--no-deps preserves the base image's CUDA-aware vLLM/torch stack).
# librosa + the deps in requirements.{common,vllm}.txt cover audio/video I/O
# for the Nemotron Nano Omni multimodal pipeline.
RUN pip install --no-cache-dir /tmp/*.whl && \
    pip install --no-cache-dir librosa \
        -r /workspace/dynamo/container/deps/requirements.common.txt \
        -r /workspace/dynamo/container/deps/requirements.vllm.txt && \
    cd /workspace/dynamo && pip install --no-cache-dir -e . --no-deps && \
    rm -f /tmp/*.whl

# vllm/vllm-openai's default ENTRYPOINT runs `vllm serve`; reset it so the
# image behaves as a plain dynamo runtime image.
ENTRYPOINT ["/bin/bash"]
