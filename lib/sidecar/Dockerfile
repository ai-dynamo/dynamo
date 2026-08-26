# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG TARGETARCH

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

# Keep every workspace manifest so the committed lockfile remains valid under
# `--locked`. Placeholder library targets satisfy Cargo's workspace discovery;
# only the shared dependency sources and one engine source tree are copied into
# a backend stage. An engine source change therefore invalidates only its stage.
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY deploy/inference-gateway/ext-proc/Cargo.toml deploy/inference-gateway/ext-proc/Cargo.toml
COPY examples/router/custom-policy-example/catalog/Cargo.toml examples/router/custom-policy-example/catalog/Cargo.toml
COPY examples/router/custom-policy-example/disagg-filter-score-pick/Cargo.toml examples/router/custom-policy-example/disagg-filter-score-pick/Cargo.toml
COPY examples/router/custom-policy-example/epp/Cargo.toml examples/router/custom-policy-example/epp/Cargo.toml
COPY examples/router/custom-policy-example/simple-filter-score-pick/Cargo.toml examples/router/custom-policy-example/simple-filter-score-pick/Cargo.toml
COPY examples/router/custom-policy-example/simple-stacked-score-pick/Cargo.toml examples/router/custom-policy-example/simple-stacked-score-pick/Cargo.toml
COPY lib/backend-common/Cargo.toml lib/backend-common/Cargo.toml
COPY lib/backend-common/examples/mocker/Cargo.toml lib/backend-common/examples/mocker/Cargo.toml
COPY lib/bench/Cargo.toml lib/bench/Cargo.toml
COPY lib/bindings/c/Cargo.toml lib/bindings/c/Cargo.toml
COPY lib/bindings/python/codegen/Cargo.toml lib/bindings/python/codegen/Cargo.toml
COPY lib/data-gen/Cargo.toml lib/data-gen/Cargo.toml
COPY lib/kv-hashing/Cargo.toml lib/kv-hashing/Cargo.toml
COPY lib/kv-router/Cargo.toml lib/kv-router/Cargo.toml
COPY lib/kvbm-common/Cargo.toml lib/kvbm-common/Cargo.toml
COPY lib/kvbm-config/Cargo.toml lib/kvbm-config/Cargo.toml
COPY lib/kvbm-consolidator/Cargo.toml lib/kvbm-consolidator/Cargo.toml
COPY lib/kvbm-engine/Cargo.toml lib/kvbm-engine/Cargo.toml
COPY lib/kvbm-kernels/Cargo.toml lib/kvbm-kernels/Cargo.toml
COPY lib/kvbm-logical/Cargo.toml lib/kvbm-logical/Cargo.toml
COPY lib/kvbm-physical/Cargo.toml lib/kvbm-physical/Cargo.toml
COPY lib/llm/Cargo.toml lib/llm/Cargo.toml
COPY lib/memory/Cargo.toml lib/memory/Cargo.toml
COPY lib/mocker/Cargo.toml lib/mocker/Cargo.toml
COPY lib/mocker/servers/sglang/Cargo.toml lib/mocker/servers/sglang/Cargo.toml
COPY lib/mocker/servers/vllm/Cargo.toml lib/mocker/servers/vllm/Cargo.toml
COPY lib/rl/Cargo.toml lib/rl/Cargo.toml
COPY lib/router-plugins/catalog/Cargo.toml lib/router-plugins/catalog/Cargo.toml
COPY lib/runtime/Cargo.toml lib/runtime/Cargo.toml
COPY lib/sidecar/common/Cargo.toml lib/sidecar/common/Cargo.toml
COPY lib/sidecar/sglang/Cargo.toml lib/sidecar/sglang/Cargo.toml
COPY lib/sidecar/trtllm/Cargo.toml lib/sidecar/trtllm/Cargo.toml
COPY lib/sidecar/vllm/Cargo.toml lib/sidecar/vllm/Cargo.toml
COPY lib/tokens/Cargo.toml lib/tokens/Cargo.toml
COPY lib/truthy/Cargo.toml lib/truthy/Cargo.toml
COPY lib/sidecar/prepare-docker-workspace.sh /usr/local/bin/prepare-docker-workspace

RUN prepare-docker-workspace

# Shared local dependency closure for all three engine sidecars. Changes here
# intentionally invalidate each backend build.
COPY lib/backend-common lib/backend-common
COPY lib/bench lib/bench
COPY lib/data-gen lib/data-gen
COPY lib/kv-hashing lib/kv-hashing
COPY lib/kv-router lib/kv-router
COPY lib/llm lib/llm
COPY lib/memory lib/memory
COPY lib/mocker lib/mocker
COPY lib/rl lib/rl
COPY lib/runtime lib/runtime
COPY lib/sidecar/common lib/sidecar/common
COPY lib/tokens lib/tokens
COPY lib/truthy lib/truthy

# Each backend uses an independent layer cache while sharing Cargo downloads and
# compiled dependencies. The locked, architecture-specific target cache also
# serializes release links, which are memory intensive under the workspace's
# thin-LTO release profile.

# ---- vLLM builder -----------------------------------------------------------
FROM builder-base AS vllm-builder

COPY lib/sidecar/vllm lib/sidecar/vllm

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-vllm-sidecar \
    && strip target/release/dynamo-vllm-sidecar \
    && install -D target/release/dynamo-vllm-sidecar /out/dynamo-vllm-sidecar

# ---- SGLang builder ---------------------------------------------------------
FROM builder-base AS sglang-builder

COPY lib/sidecar/sglang lib/sidecar/sglang

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-sglang-sidecar \
    && strip target/release/dynamo-sglang-sidecar \
    && install -D target/release/dynamo-sglang-sidecar /out/dynamo-sglang-sidecar

# ---- TensorRT-LLM builder ---------------------------------------------------
FROM builder-base AS trtllm-builder

COPY lib/sidecar/trtllm lib/sidecar/trtllm

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=dynamo-sidecar-cargo-git,target=/usr/local/cargo/git \
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
