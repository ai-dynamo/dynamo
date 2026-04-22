# syntax=docker/dockerfile:1.6
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run the rotating_tokens Rust backend.
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

# BuildKit cache mount speeds up incremental rebuilds by persisting the
# cargo registry (crates.io index + downloaded crate sources) across builds.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build -p dynamo-rotating-tokens --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/dynamo-rotating-tokens /usr/local/bin/

ENTRYPOINT ["dynamo-rotating-tokens"]
