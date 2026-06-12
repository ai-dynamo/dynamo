# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo P/D routing sidecar (vanilla vLLM disaggregation).
# 2-stage build: pure-Rust binary -> minimal Ubuntu runtime.
#
# Build via buildx with the repo root as the `dynamo` build context:
#   docker buildx build --build-context dynamo=../../.. \
#     -t dynamo/pd-sidecar:dev -f Dockerfile .

ARG RUST_IMAGE=rust:1.93.1
ARG BASE_IMAGE=ubuntu:24.04

# =============================================================================
# Stage 1: Build the sidecar binary
# =============================================================================
FROM ${RUST_IMAGE} AS builder

WORKDIR /dynamo

# Copy workspace manifests + lockfile. The Cargo workspace refuses to load if a
# listed member is missing, so we copy the full member tree below; only
# `dynamo-pd-sidecar` (a pure-Rust crate, no dynamo deps) is actually compiled.
COPY --from=dynamo .cargo/ .cargo/
COPY --from=dynamo Cargo.toml Cargo.lock README.md ./
COPY --from=dynamo lib/ lib/
COPY --from=dynamo deploy/inference-gateway/ deploy/inference-gateway/

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release -p dynamo-pd-sidecar \
    && cp target/release/dynamo-pd-sidecar /dynamo-pd-sidecar

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY --from=builder /dynamo-pd-sidecar /dynamo-pd-sidecar

RUN useradd -r -u 65532 -g nogroup nonroot
USER 65532:65534

ENTRYPOINT ["/dynamo-pd-sidecar"]
