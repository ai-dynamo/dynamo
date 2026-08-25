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

# Release linking is memory intensive: every sidecar links the same large
# dependency graph under `lto = "thin"` and `codegen-units = 1`. Build the three
# targets in sequence so only one link runs at a time on smaller builders, while
# shared dependencies still compile in parallel during the first invocation.
# Building each package on its own also matches what `cargo build -p <crate>`
# produces locally: a single invocation would unify Cargo features across all
# three, so a feature added to one sidecar would silently reach the other two.
#
# There is no dependency-cache layer, so a source change recompiles the
# dependencies — acceptable for this occasional image build.
RUN cargo build --release --locked -p dynamo-vllm-sidecar \
    && cargo build --release --locked -p dynamo-sglang-sidecar \
    && cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip \
        target/release/dynamo-vllm-sidecar \
        target/release/dynamo-sglang-sidecar \
        target/release/dynamo-trtllm-sidecar

# ---- runtime ----------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# ca-certificates supports tokenizer downloads and TLS connections to Dynamo
# services. TLS crypto is statically linked, so no engine or GPU libraries are
# needed in this image. The sidecar user is created with a home directory
# because the Hugging Face tokenizer cache lives there.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder \
    /src/target/release/dynamo-vllm-sidecar \
    /src/target/release/dynamo-sglang-sidecar \
    /src/target/release/dynamo-trtllm-sidecar \
    /usr/local/bin/

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 1000

# Dynamo system endpoints (/live, /health) — the operator sets DYN_SYSTEM_PORT=9090.
EXPOSE 9090

# Intentionally no ENTRYPOINT. Kubernetes deployments select one of the three
# engine-specific binaries with `command`.
