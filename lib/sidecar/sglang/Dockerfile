# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal image for the SGLang Rust sidecar (`dynamo-sglang-sidecar`).
# A pure gRPC connector — no GPU, no engine runtime — that runs beside the
# engine container over loopback (see deploy/agg.yaml).
#
#   docker build --secret id=buf_token,env=BUF_TOKEN \
#     -f lib/sidecar/sglang/Dockerfile -t dynamo-sglang-sidecar:1.3.0 .

FROM rust:1.96.1-bookworm AS builder
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

RUN --mount=type=secret,id=buf_token,required=true \
    CARGO_REGISTRIES_BUF_TOKEN="Bearer $(cat /run/secrets/buf_token)" \
    cargo build --release --locked -p dynamo-sglang-sidecar \
    && strip target/release/dynamo-sglang-sidecar

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group sidecar

COPY --from=builder /src/target/release/dynamo-sglang-sidecar /usr/local/bin/dynamo-sglang-sidecar

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 1000
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/dynamo-sglang-sidecar"]
