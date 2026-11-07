# syntax=docker/dockerfile:1.10
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build context: the Dynamo checkout combining affinity and router registration.
FROM rust:1.96.1-trixie AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config protobuf-compiler libclang-dev libssl-dev \
    python3 python3-venv patchelf curl unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . .
# Bound compilation memory for the development builder.
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=0 \
    CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
ARG TARGETARCH
FROM builder AS binaries
COPY --from=build-assets swagger-ui-v5.33.1.zip /opt/build-assets/swagger-ui-v5.33.1.zip
# Use the same pinned Swagger archive without a build-time GitHub download.
RUN python3 -c 'from pathlib import Path; p=Path(".cargo/config.toml"); s=p.read_text(); old="https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.33.1.zip"; assert old in s; p.write_text(s.replace(old, "file:///opt/build-assets/swagger-ui-v5.33.1.zip"))'
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=gr-runtime-target-${TARGETARCH},target=/workspace/target,sharing=locked \
    cargo build --locked --release \
    -p dynamo-global-discovery-service -p dynamo-global-frontend \
    && mkdir -p /artifacts \
    && cp target/release/global-discovery-service target/release/global-frontend /artifacts/

FROM debian:trixie-slim AS binary-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libssl3t64 libzmq5 \
    && rm -rf /var/lib/apt/lists/*
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/ai-dynamo/dynamo" \
    org.opencontainers.image.revision="$SOURCE_REVISION"
USER 10001:0
FROM binary-runtime AS global-discovery-service
COPY --from=binaries /artifacts/global-discovery-service /usr/local/bin/global-discovery-service
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/global-discovery-service"]
FROM binary-runtime AS global-frontend
COPY --from=binaries /artifacts/global-frontend /usr/local/bin/global-frontend
EXPOSE 8000 50051
ENTRYPOINT ["/usr/local/bin/global-frontend"]

FROM binaries AS bindings
ENV CARGO_TARGET_DIR=/workspace/target
RUN apt-get update && apt-get install -y --no-install-recommends libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --no-cache-dir 'maturin>=1,<2'
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=gr-runtime-target-${TARGETARCH},target=/workspace/target,sharing=locked \
    /opt/build-venv/bin/maturin build --locked --release \
    --manifest-path lib/bindings/python/Cargo.toml \
    --features kv-indexer,slot-tracker,select-service \
    --interpreter /opt/build-venv/bin/python --auditwheel skip --out /wheels

FROM python:3.13-slim-trixie AS pool-relay
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libssl3t64 libzmq5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=bindings /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels
# The wheel supplies dynamo.runtime and dynamo.llm; the component is Python.
COPY components/src/dynamo/pool_relay /opt/dynamo/components/dynamo/pool_relay
ENV PYTHONPATH=/opt/dynamo/components PYTHONUNBUFFERED=1
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/ai-dynamo/dynamo" \
    org.opencontainers.image.revision="$SOURCE_REVISION"
USER 10001:0
EXPOSE 50051 9090 9100
ENTRYPOINT ["python3", "-m", "dynamo.pool_relay"]
