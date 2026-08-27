# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only image containing the vLLM, SGLang, and TensorRT-LLM Rust sidecars.
# The dynamo-sidecar entrypoint selects an engine-specific binary; the GPU
# engine continues to run in a separate container.
#
# Build from the repository root:
#   docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.3.0 .

# Use NVIDIA's public, signed, multi-architecture Ubuntu base for both build
# and runtime stages. The sidecars are CPU-only and need no framework or CUDA
# runtime image.
ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20250619

# ---- common builder ---------------------------------------------------------
FROM ${BASE_IMAGE} AS builder-base

ARG TARGETARCH

# Superset of the build dependencies required by all sidecar crates:
# aws-lc-sys/ring need cmake and a C toolchain (plus perl); bindgen needs
# libclang; tonic-build (build.rs) needs protoc, and libprotobuf-dev supplies
# the well-known protos the engine .proto files import (vllm_grpc.proto pulls
# in google/protobuf/struct.proto); utoipa-swagger-ui downloads its pinned web
# asset through the system curl binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        clang \
        curl \
        git \
        libclang-dev \
        protobuf-compiler \
        libprotobuf-dev \
        perl \
        pkg-config \
        wget \
    && rm -rf /var/lib/apt/lists/*

# NGC does not publish a Rust toolchain image. Install the same pinned Rust
# release used by the workspace through the pinned rustup installer.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    RUST_VERSION=1.96.1

RUN case "${TARGETARCH}" in \
        amd64) RUST_TARGET=x86_64-unknown-linux-gnu ;; \
        arm64) RUST_TARGET=aarch64-unknown-linux-gnu ;; \
        *) echo "Unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && wget --tries=3 --waitretry=5 \
        "https://static.rust-lang.org/rustup/archive/1.28.1/${RUST_TARGET}/rustup-init" \
    && chmod +x rustup-init \
    && ./rustup-init -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" --default-host "${RUST_TARGET}" \
    && rm rustup-init

WORKDIR /src

# Cargo resolves every sidecar through the root workspace and their shared
# local crates. Copy the workspace inputs once instead of maintaining a
# duplicated transitive dependency list for each engine.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .cargo/ .cargo/
COPY lib/ lib/
COPY examples/router/custom-policy-example/ examples/router/custom-policy-example/
COPY deploy/inference-gateway/ext-proc/ deploy/inference-gateway/ext-proc/

# ---- vLLM builder -----------------------------------------------------------
FROM builder-base AS vllm-builder

ARG TARGETARCH

# The three builders share one target cache per architecture, so each backend
# reuses the dependency artifacts the previous one compiled. sharing=locked
# serializes them, which is required for correctness — cargo's .package-cache
# lock lives in $CARGO_HOME, outside these mounts, so it cannot coordinate
# separate build containers — and is also what bounds peak memory, since
# codegen-units=1 plus LTO makes each release link expensive on its own.
# Every id is per-architecture so the amd64 and arm64 halves of a multi-platform
# build do not serialize against each other.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-vllm-sidecar \
    && strip target/release/dynamo-vllm-sidecar \
    && install -D target/release/dynamo-vllm-sidecar /out/dynamo-vllm-sidecar

# ---- SGLang builder ---------------------------------------------------------
FROM builder-base AS sglang-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-sglang-sidecar \
    && strip target/release/dynamo-sglang-sidecar \
    && install -D target/release/dynamo-sglang-sidecar /out/dynamo-sglang-sidecar

# ---- TensorRT-LLM builder ---------------------------------------------------
FROM builder-base AS trtllm-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar \
    && install -D target/release/dynamo-trtllm-sidecar /out/dynamo-trtllm-sidecar

# ---- runtime ----------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

# ca-certificates supports tokenizer downloads and TLS connections to Dynamo
# services. TLS crypto is statically linked, so no engine or GPU libraries are
# needed in this image. The dynamo user uses group 0 for OpenShift
# compatibility and has a home directory for the Hugging Face tokenizer cache.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && userdel ubuntu \
    && useradd --create-home --uid 1000 --gid 0 dynamo

COPY --from=vllm-builder /out/dynamo-vllm-sidecar /usr/local/bin/
COPY --from=sglang-builder /out/dynamo-sglang-sidecar /usr/local/bin/
COPY --from=trtllm-builder /out/dynamo-trtllm-sidecar /usr/local/bin/

COPY --chmod=755 lib/sidecar/dynamo-sidecar /usr/local/bin/dynamo-sidecar

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 1000

# Dynamo system endpoints (/live, /health) — the operator sets DYN_SYSTEM_PORT=9090.
EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/dynamo-sidecar"]
CMD ["--help"]
