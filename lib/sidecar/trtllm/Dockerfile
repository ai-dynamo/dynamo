# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal image for the TensorRT-LLM Rust sidecar (`dynamo-trtllm-sidecar`).
# A pure gRPC connector — no GPU, no engine runtime — that runs beside the
# engine container over loopback (see deploy/agg.yaml).
#
#   docker build -f lib/sidecar/trtllm/Dockerfile -t dynamo-trtllm-sidecar:1.3.0 .

# CI sets DOCKER_PROXY to the authenticated Docker Hub mirror. It remains empty
# for the documented local build command.
ARG DOCKER_PROXY=

FROM ${DOCKER_PROXY}library/rust:1.96.1-bookworm AS builder
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

RUN cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar

FROM ${DOCKER_PROXY}library/debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder /src/target/release/dynamo-trtllm-sidecar /usr/local/bin/dynamo-trtllm-sidecar

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 1000
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/dynamo-trtllm-sidecar"]
