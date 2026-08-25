# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only image containing the vLLM, SGLang, and TensorRT-LLM Rust sidecars.
# Each deployment selects its engine-specific binary through the container
# command; the GPU engine continues to run in a separate container.
#
# Build from the repository root:
#   docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.3.0 .

# ---- builder ----------------------------------------------------------------
FROM rust:1.96.1-bookworm AS builder

# Superset of the build dependencies required by all sidecar crates.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        clang \
        libclang-dev \
        protobuf-compiler \
        libprotobuf-dev \
        perl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Release linking is memory intensive. Limit Cargo to one rustc process so the
# three binaries are linked serially instead of exhausting smaller builders.
RUN cargo build --release --locked --jobs 1 \
        -p dynamo-vllm-sidecar \
        -p dynamo-sglang-sidecar \
        -p dynamo-trtllm-sidecar \
    && strip \
        target/release/dynamo-vllm-sidecar \
        target/release/dynamo-sglang-sidecar \
        target/release/dynamo-trtllm-sidecar

# ---- runtime ----------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# ca-certificates supports tokenizer downloads and TLS connections to Dynamo
# services. TLS crypto is statically linked, so no engine or GPU libraries are
# needed in this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder \
    /src/target/release/dynamo-vllm-sidecar \
    /src/target/release/dynamo-sglang-sidecar \
    /src/target/release/dynamo-trtllm-sidecar \
    /usr/local/bin/

USER 1000
EXPOSE 9090

# Intentionally no ENTRYPOINT. Kubernetes deployments select one of the three
# engine-specific binaries with `command`.
