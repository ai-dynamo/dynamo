# syntax=docker/dockerfile:1.6
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run the mocker Rust backend — a full-fidelity mocked LLM
# engine wrapped in the `LLMEngine` contract from `dynamo-backend-common`.
#
# Build context MUST be the dynamo workspace root so Cargo can resolve all
# workspace crates — see docker-compose.yml in this directory, which sets
# `context: ../../../..` for this purpose.

FROM rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        libclang-dev \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# BuildKit cache mounts persist across builds:
#   /usr/local/cargo/registry — downloaded crate sources (avoids redownloading)
#   /build/target             — compiled artifacts (avoids full recompile)
# Cache mounts are NOT part of the resulting image, so copy the binary out
# of the cache to /out/ within the same RUN so later stages can COPY it.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    mkdir -p /out \
    && cargo build -p dynamo-mocker-backend --release \
    && cp /build/target/release/dynamo-mocker-backend /out/

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/dynamo-mocker-backend /usr/local/bin/

ENTRYPOINT ["dynamo-mocker-backend"]
