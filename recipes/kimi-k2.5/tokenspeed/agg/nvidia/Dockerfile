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
#     --build-arg BASE_IMAGE=<your-tokenspeed-image>:<tag> \
#     --target dev \
#     -t <your-registry>/dynamo-tokenspeed:dev \
#     .
#
#   # 2. Slim runtime image (no compilers / no Rust toolchain):
#   docker build \
#     -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
#     --build-arg BASE_IMAGE=<your-tokenspeed-image>:<tag> \
#     --target runtime \
#     -t <your-registry>/dynamo-tokenspeed:runtime \
#     .
#
# Use --build-arg ARCH=arm64 for arm64 base images.
ARG BASE_IMAGE=tokenspeed:local
ARG ARCH=amd64
ARG CARGO_BUILD_JOBS=16

FROM ${BASE_IMAGE} AS tokenspeed_base
ARG ARCH
ENV PIP_BREAK_SYSTEM_PACKAGES=1
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*

# Dynamo services used by local agg/backend smoke tests.
RUN set -eux; \
    curl -fsSL "https://github.com/etcd-io/etcd/releases/download/v3.6.7/etcd-v3.6.7-linux-${ARCH}.tar.gz" | \
        tar xz -C /usr/bin --strip-components=1 \
            "etcd-v3.6.7-linux-${ARCH}/etcd" \
            "etcd-v3.6.7-linux-${ARCH}/etcdctl"; \
    curl -fsSL "https://github.com/nats-io/nats-server/releases/download/v2.12.4/nats-server-v2.12.4-linux-${ARCH}.tar.gz" | \
        tar xz -C /usr/bin --strip-components=1 \
            "nats-server-v2.12.4-linux-${ARCH}/nats-server"

FROM tokenspeed_base AS build_tools
ARG CARGO_BUILD_JOBS
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
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
