# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo Rust EPP Dockerfile
# 2-stage build: Rust binary → Ubuntu runtime
#
# Unlike the Go EPP which requires Rust → Go CGO linking, this is a pure
# Rust binary with no CGO bridge. Single compile step.
#
# Build via Makefile (recommended):
#   make image-load
#
# Build manually:
#   docker buildx build \
#     --build-context dynamo=../../.. \
#     -t dynamo/dynamo-rust-epp:dev .

ARG RUST_IMAGE=rust:1.93.1
ARG BASE_IMAGE=ubuntu:24.04

# =============================================================================
# Stage 1: Build Rust EPP binary
# =============================================================================
FROM ${RUST_IMAGE} AS builder

ARG TARGETARCH

# sccache configuration (matches Go EPP pattern)
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# protobuf-compiler provides protoc; libprotobuf-dev provides the well-known
# .proto files (google/protobuf/struct.proto etc.) that our vendored Envoy
# protos import. libclang-dev is needed by bindgen (nixl-sys transitive dep).
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /dynamo

# Install sccache using the shared helper
COPY --from=dynamo container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}

# Copy Cargo workspace manifests, lockfile, and README
COPY --from=dynamo .cargo/ .cargo/
COPY --from=dynamo Cargo.toml Cargo.lock README.md ./

# Copy workspace crates that dynamo-ext-proc depends on
COPY --from=dynamo lib/ lib/

# Copy the ext-proc crate itself
COPY --from=dynamo deploy/inference-gateway/ext-proc/ deploy/inference-gateway/ext-proc/

# Build the binary
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETARCH} \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETARCH} \
    --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-rust-epp-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cargo build --release -p dynamo-ext-proc && \
    cp target/release/dynamo-ext-proc /dynamo-ext-proc && \
    if [ "$USE_SCCACHE" = "true" ]; then /tmp/use-sccache.sh show-stats "dynamo-ext-proc" || true; fi

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY --from=builder /dynamo-ext-proc /dynamo-ext-proc

RUN useradd -r -u 65532 -g nogroup nonroot
USER 65532:65534

ENTRYPOINT ["/dynamo-ext-proc"]
