# syntax=docker/dockerfile:1.10.0-labs
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TokenSpeed bring-up image for iterating on the Dynamo common LLMEngine path.
# Layers Dynamo (built from source via maturin) on top of a public TokenSpeed
# runtime image, so you can ship a Dynamo+TokenSpeed worker before NVIDIA
# publishes an `nvcr.io/nvidia/ai-dynamo/tokenspeed-runtime` image.
#
# Build context must be the Dynamo repo root (the source tree is COPY'd in).
#
# Examples:
#   # 1. Dev image (keeps the Rust toolchain for in-pod rebuilds):
#   docker build \
#     -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
#     --target dev \
#     -t <your-registry>/dynamo-tokenspeed:dev \
#     .
#
#   # 2. Slim runtime image (no compilers / no Rust toolchain):
#   docker build \
#     -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
#     --target runtime \
#     -t <your-registry>/dynamo-tokenspeed:runtime \
#     .
#
# The default BASE_IMAGE pins lightseekorg/tokenspeed-runner at a CUDA-13
# tag (matches B200's CUDA 13.x driver, ~4.66 GB). To target a different
# TokenSpeed runtime build, override it:
#   --build-arg BASE_IMAGE=docker.io/lightseekorg/tokenspeed-runner:cu129-torch-2.11.0
# or for arm64 base images:
#   --build-arg ARCH=arm64
#
# The runner image is the runtime base only; this Dockerfile installs the
# TokenSpeed engine itself from github.com/lightseekorg/tokenspeed, following
# the upstream guide at lightseek.org/tokenspeed/guides/getting-started. Pin
# the TokenSpeed source ref via --build-arg TOKENSPEED_GIT_REF=<ref>.
ARG BASE_IMAGE=docker.io/lightseekorg/tokenspeed-runner:cu130-torch-2.11.0
ARG ARCH=amd64
ARG CARGO_BUILD_JOBS=16

FROM ${BASE_IMAGE} AS tokenspeed_base
ARG ARCH
ARG TOKENSPEED_GIT_REF=main
ENV PIP_BREAK_SYSTEM_PACKAGES=1
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git; \
    rm -rf /var/lib/apt/lists/*

# Install TokenSpeed from upstream source. The public lightseekorg/tokenspeed-runner
# image ships only the runtime base (CUDA, PyTorch, mooncake); the engine itself
# is not yet published on PyPI. Mirrors the upstream guide at
# https://lightseek.org/tokenspeed/guides/getting-started.
# Override the source ref (a branch / tag / commit SHA) with --build-arg TOKENSPEED_GIT_REF=<ref>.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    git clone https://github.com/lightseekorg/tokenspeed.git /opt/tokenspeed; \
    cd /opt/tokenspeed; \
    git checkout "${TOKENSPEED_GIT_REF}"; \
    pip install -e "./python" --no-build-isolation; \
    pip install -e tokenspeed-kernel/python/ --no-build-isolation; \
    pip install -e tokenspeed-scheduler/

# NOTE: discovery uses the in-cluster Kubernetes API (`--discovery-backend kubernetes`)
# and the event-plane connects to a separate `nats:2.12.4` Deployment, so the worker
# image doesn't need the etcd or nats-server binaries. Add them back if you need to
# run a single-pod 4-process smoke test outside Kubernetes.

FROM tokenspeed_base AS build_tools
ARG CARGO_BUILD_JOBS
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"
# build-essential and git already installed in tokenspeed_base for the
# TokenSpeed kernel build; this layer adds the remaining Rust/Dynamo tooling.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libclang-dev \
        pkg-config \
        protobuf-compiler; \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    set -eux; \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl -fsSL https://sh.rustup.rs | \
            sh -s -- -y --profile minimal --default-toolchain stable; \
    fi; \
    cargo install maturin --locked

FROM build_tools AS wheel_builder
WORKDIR /workspace/dynamo
COPY . /workspace/dynamo
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    set -eux; \
    cd /workspace/dynamo/lib/bindings/python; \
    maturin build --release -o /tmp/dynamo-dist

FROM tokenspeed_base AS runtime
COPY --from=wheel_builder /workspace/dynamo /workspace/dynamo
COPY --from=wheel_builder /tmp/dynamo-dist/*.whl /tmp/
RUN set -eux; \
    python3 -m pip install --no-cache-dir /tmp/*.whl; \
    cd /workspace/dynamo; \
    python3 -m pip install --no-cache-dir -e . --no-deps; \
    rm -f /tmp/*.whl
WORKDIR /workspace/dynamo

FROM build_tools AS dev
COPY --from=wheel_builder /workspace/dynamo /workspace/dynamo
COPY --from=wheel_builder /tmp/dynamo-dist/*.whl /tmp/
RUN set -eux; \
    python3 -m pip install --no-cache-dir /tmp/*.whl; \
    cd /workspace/dynamo; \
    python3 -m pip install --no-cache-dir -e . --no-deps; \
    rm -f /tmp/*.whl
WORKDIR /workspace/dynamo
