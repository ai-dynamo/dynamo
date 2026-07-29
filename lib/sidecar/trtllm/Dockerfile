# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal image for the TensorRT-LLM Rust sidecar (`dynamo-trtllm-sidecar`).
#
# A pure gRPC connector: no GPU, no TensorRT-LLM runtime, so it ships in a slim
# Debian image and runs beside the engine container over loopback (deploy/agg.yaml).
#
# Build from the repository root:
#   docker build -f lib/sidecar/trtllm/Dockerfile -t dynamo-trtllm-sidecar:1.3.0 .

# ---- builder ----------------------------------------------------------------
# Workspace toolchain (rust-toolchain.toml: 1.96.1).
FROM rust:1.96.1-bookworm AS builder

# aws-lc-sys/ring: cmake + C toolchain (+perl); bindgen: libclang; tonic-build:
# protoc. gcc/make/git ship with the rust image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        clang \
        libclang-dev \
        protobuf-compiler \
        perl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# --locked pins the checked-in Cargo.lock. No dep-cache layer, so a source change
# recompiles deps — fine for this occasional build.
RUN cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar

# ---- runtime (minimal) ------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# ca-certificates: HTTPS to Hugging Face for the tokenizer + rustls TLS to
# etcd/NATS. TLS crypto (aws-lc-rs) is static, so no libssl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder /src/target/release/dynamo-trtllm-sidecar /usr/local/bin/dynamo-trtllm-sidecar

# Numeric UID so Kubernetes runAsNonRoot can verify it; the home dir holds the
# Hugging Face tokenizer cache.
USER 1000
# Dynamo system endpoints (/live, /health); operator sets DYN_SYSTEM_PORT=9090.
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/dynamo-trtllm-sidecar"]
