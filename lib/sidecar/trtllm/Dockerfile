# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal image for the TensorRT-LLM Rust sidecar (`dynamo-trtllm-sidecar`).
#
# The sidecar is a pure gRPC connector + tokenizer; it needs no GPU and no
# TensorRT-LLM runtime, so it ships in a slim Debian image separate from the
# heavy TRT-LLM engine container. The two run side by side in one pod and talk
# over loopback (see deploy/agg.yaml).
#
# Build from the repository root (the workspace is needed to compile the crate):
#   docker build -f lib/sidecar/trtllm/Dockerfile -t dynamo-trtllm-sidecar:1.3.0 .

# ---- builder ----------------------------------------------------------------
# Pinned to the workspace toolchain (rust-toolchain.toml: 1.96.1).
FROM rust:1.96.1-bookworm AS builder

# aws-lc-sys/ring need cmake + a C toolchain (+perl); bindgen needs libclang;
# tonic-build (build.rs) needs protoc. gcc/make/git come with the rust image.
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

# Build only the sidecar binary; --locked pins the checked-in Cargo.lock. There
# is no dependency-cache layer, so a source change recompiles deps — acceptable
# for this occasional image build.
RUN cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar

# ---- runtime (minimal) ------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# ca-certificates: HTTPS to Hugging Face for the tokenizer, and rustls TLS to
# etcd/NATS. TLS crypto (aws-lc-rs) is statically linked, so no libssl needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder /src/target/release/dynamo-trtllm-sidecar /usr/local/bin/dynamo-trtllm-sidecar

# Numeric UID (not the name) so Kubernetes runAsNonRoot can verify it; the home
# dir created above holds the Hugging Face tokenizer cache.
USER 1000
# Dynamo system endpoints (/live, /health) — the operator sets DYN_SYSTEM_PORT=9090.
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/dynamo-trtllm-sidecar"]
