# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only image containing the vLLM, SGLang, and TensorRT-LLM Rust sidecars.
# The dynamo-sidecar entrypoint selects an engine-specific binary; the GPU
# engine continues to run in a separate container.
#
# Build from the repository root:
#   docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.3.0 .

# ---- common builder ---------------------------------------------------------
FROM rust:1.96.1-bookworm AS builder-base

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

# ---- vLLM builder -----------------------------------------------------------
FROM builder-base AS vllm-builder

ARG TARGETARCH

# sharing=locked: cargo's .package-cache lock lives outside these mounts, so it
# cannot serialize the concurrent cargo processes a multi-platform build starts.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-vllm-sidecar \
    && strip target/release/dynamo-vllm-sidecar \
    && install -D target/release/dynamo-vllm-sidecar /out/dynamo-vllm-sidecar

# ---- SGLang builder ---------------------------------------------------------
FROM builder-base AS sglang-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-sglang-sidecar \
    && strip target/release/dynamo-sglang-sidecar \
    && install -D target/release/dynamo-sglang-sidecar /out/dynamo-sglang-sidecar

# ---- TensorRT-LLM builder ---------------------------------------------------
FROM builder-base AS trtllm-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar \
    && install -D target/release/dynamo-trtllm-sidecar /out/dynamo-trtllm-sidecar

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
