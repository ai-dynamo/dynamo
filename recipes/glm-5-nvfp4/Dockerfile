# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE="lmsysorg/sglang:v0.5.10.post1-cu130"
ARG ARCH=arm64
ARG DYNAMO_COMMIT="main"
ARG CARGO_BUILD_JOBS="16"

# ======== Stage 1: Builder (compiles Rust, builds wheel) ========
FROM ${BASE_IMAGE} AS builder

ARG DYNAMO_COMMIT
ARG CARGO_BUILD_JOBS
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}

WORKDIR /build

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential pkg-config libclang-dev protobuf-compiler git; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

ENV CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN cargo install maturin --locked

RUN git clone https://github.com/ai-dynamo/dynamo.git /build/dynamo && \
    cd /build/dynamo && git checkout ${DYNAMO_COMMIT}

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cd /build/dynamo/lib/bindings/python && \
    maturin build --release && \
    mkdir -p /build/dist && \
    cp target/wheels/*.whl /build/dist/

# ======== Stage 2: Final image ========
FROM ${BASE_IMAGE}
ARG ARCH=arm64

# Install flashinfer with matching jit-cache.
# Pin to 0.6.7 and install jit-cache from flashinfer wheel index so all three
# packages (python, cubin, jit-cache) are consistent.
RUN pip install "flashinfer-python==0.6.7" "flashinfer-cubin==0.6.7" && \
    pip install "flashinfer-jit-cache==0.6.7" --index-url https://flashinfer.ai/whl/cu130

# Install dynamo from source (includes _compat.py which handles sglang API changes)
COPY --from=builder /build/dynamo /sgl-workspace/dynamo
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    cd /sgl-workspace/dynamo && \
    pip install --no-cache-dir -e ".[sglang]" --no-deps && \
    rm -f /tmp/*.whl

