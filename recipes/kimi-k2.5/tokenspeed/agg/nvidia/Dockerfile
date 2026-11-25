# syntax=docker/dockerfile:1.10.0-labs
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TokenSpeed bring-up image for iterating on the Dynamo common LLMEngine path.
# Layers Dynamo (built from source via maturin) on top of TokenSpeed's runner
# image plus pinned TokenSpeed nightly wheels, so you can ship a
# Dynamo+TokenSpeed worker before NVIDIA publishes an
# `nvcr.io/nvidia/ai-dynamo/tokenspeed-runtime` image.
#
# Unlike Dynamo's vLLM and SGLang images, this does not start FROM the engine's
# own published container: no published TokenSpeed image contains the pinned
# revision. See "Why this build differs from the vLLM and SGLang images" in
# this directory's README.md.
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
# The default BASE_IMAGE pins lightseekorg/tokenspeed-runner at an immutable
# CUDA-13 multi-arch digest (matches B200/GB300's CUDA 13.x driver). The
# TokenSpeed wheels follow the target platform (BuildKit's TARGETARCH), so a
# build on an arm64 host (GB200/GB300) needs no extra flags. --build-arg
# ARCH=amd64|arm64 overrides it.
#
# The runner image is the runtime base only. This Dockerfile installs the
# TokenSpeed nightly wheels that SemiAnalysisAI/InferenceX#4780 pins: the
# Python runtime, kernel, and Rust engine (`tokenspeed_engine`, which carries
# the msgpack wire frontend the Dynamo worker uses) built from upstream
# revision 0e6ee353d7bd, with scheduler 0.1.49 and FlashInfer 0.7.2.post1.
# Each wheel is pinned by SHA256. The Dynamo worker drives the headless engine
# over that wire, so a TokenSpeed older than this rewrite (any revision that
# still ships an in-process `Engine` with a `tokenizer_manager`) cannot work.
ARG BASE_IMAGE=ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.1rc2@sha256:1086943cd27a22b7d414c64ca088bee24f53a971fdae78e231c182c50b4e43d2
ARG ARCH=
ARG CARGO_BUILD_JOBS=16

FROM ${BASE_IMAGE} AS tokenspeed_base
ARG ARCH
ARG TARGETARCH
ARG TOKENSPEED_WHL=https://github.com/lightseekorg/whl/releases/download
ARG TOKENSPEED_VERSION=0.1.1.post20261102
ARG TOKENSPEED_KERNEL_VERSION=0.1.4.post20261102
ARG TOKENSPEED_ENGINE_VERSION=0.1.0.post20261102
ARG TOKENSPEED_SCHEDULER_VERSION=0.1.49
ARG TOKENSPEED_FLASHINFER_VERSION=0.7.2.post1
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

# Install the pinned TokenSpeed nightly wheels (CPython 3.12, as the runner
# ships), then the matching precompiled FlashInfer kernels so startup does not
# rebuild them. The SHA256 pins cover the three wheels for each architecture;
# change them together with the versions above.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    ARCH="${ARCH:-${TARGETARCH}}"; \
    case "${ARCH}" in \
        amd64) \
            KERNEL_PLATFORM=manylinux_2_28_x86_64; ENGINE_PLATFORM=linux_x86_64; \
            KERNEL_SHA256=6c9ac7ad2e2295b45b84168dd77de43aef8d3e1c39a285784ae07c232c70249f; \
            ENGINE_SHA256=ec830b44ea494ee4942e5db97b650466db9e9d63970cee1824a0f5c9447d00af ;; \
        arm64) \
            KERNEL_PLATFORM=manylinux_2_28_aarch64; ENGINE_PLATFORM=linux_aarch64; \
            KERNEL_SHA256=efaa5239fb7bbd55374ce8af056b9ca7b113e591d71ad480c084d0262656fdcb; \
            ENGINE_SHA256=c4fbfdbda25db1495a01431d3e115da235faa94b1050c2558762f5f468402b2e ;; \
        *) echo "unsupported ARCH=${ARCH}" >&2; exit 1 ;; \
    esac; \
    python3 -m pip install --upgrade --index-url https://pypi.org/simple \
        "${TOKENSPEED_WHL}/tokenspeed-nightly-v${TOKENSPEED_VERSION}/tokenspeed-${TOKENSPEED_VERSION}-py3-none-any.whl#sha256=82ada30115b34ba4580305e31d0bcb5c790cf319beef7cbe6deb855490e6590d" \
        "${TOKENSPEED_WHL}/tokenspeed-kernel-nightly-v${TOKENSPEED_KERNEL_VERSION}-cu130/tokenspeed_kernel-${TOKENSPEED_KERNEL_VERSION}-cp312-cp312-${KERNEL_PLATFORM}.whl#sha256=${KERNEL_SHA256}" \
        "${TOKENSPEED_WHL}/tokenspeed-engine-nightly-v${TOKENSPEED_ENGINE_VERSION}/tokenspeed_engine-${TOKENSPEED_ENGINE_VERSION}-cp312-cp312-${ENGINE_PLATFORM}.whl#sha256=${ENGINE_SHA256}" \
        "tokenspeed-scheduler==${TOKENSPEED_SCHEDULER_VERSION}"; \
    python3 -m flashinfer download-kernels \
        --cuda-version 13.0 --flashinfer-version "${TOKENSPEED_FLASHINFER_VERSION}"

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
# libprotobuf-dev ships google/protobuf/*.proto, needed by the sidecar protos.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libclang-dev \
        pkg-config \
        protobuf-compiler \
        libprotobuf-dev; \
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
