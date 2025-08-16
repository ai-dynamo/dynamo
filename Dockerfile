# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASEIMAGE=ubuntu:24.04
FROM ${BASEIMAGE} as uv-installer
RUN apt-get update && apt-get install -y curl
ARG UV_VERSION=0.8.3
# Download uv binary and verify against official checksum
RUN cd /tmp && \
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" -o uv.tar.gz && \
    echo "$(curl -fsSL https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz.sha256 | cut -d' ' -f1)  uv.tar.gz" | sha256sum -c
RUN cd /tmp && tar -xzf uv.tar.gz && mv uv-*/uv /usr/local/bin/

# Base image with common dependencies
FROM ${BASEIMAGE} as dynamo-base
COPY --from=uv-installer /usr/local/bin/uv /bin/uv
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-dev python3-pip python3-venv libucx0 curl \
    build-essential git pkg-config cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CARGO_BUILD_JOBS=16
RUN mkdir /opt/dynamo && \
    uv venv /opt/dynamo/venv --python 3.12 && \
    . /opt/dynamo/venv/bin/activate && \
    uv pip install pip

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Rust build environment
FROM dynamo-base as rust-base
RUN apt update -y && \
    apt install --no-install-recommends -y \
    wget protobuf-compiler libssl-dev libclang-dev

# Install CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    apt install -y ./cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt install -y cuda-toolkit-12-8 && \
    rm cuda-keyring_1.1-1_all.deb

ENV CUDA_COMPUTE_CAP=80
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV CUDA_ROOT=/usr/local/cuda-12.8
ENV CUDA_PATH=/usr/local/cuda-12.8
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
ENV RUST_VERSION=1.87.0

RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/x86_64-unknown-linux-gnu/rustup-init" && \
    echo "a3339fb004c3d0bb9862ba0bce001861fe5cbde9c10d16591eb3f39ee6cd3e7f *rustup-init" | sha256sum -c - && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.87.0 --default-host x86_64-unknown-linux-gnu && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Build stage
FROM rust-base as builder
WORKDIR /workspace
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY components/ components/
COPY lib/ lib/
COPY launch/ launch/
COPY deploy/ deploy/

ENV CARGO_TARGET_DIR=/workspace/target
RUN cargo build --release --locked --features llamacpp,cuda

# Build Python wheels
RUN mkdir -p /workspace/deploy/sdk/src/dynamo/sdk/cli/bin/ && \
    rm -f /workspace/deploy/sdk/src/dynamo/sdk/cli/bin/* && \
    ln -sf /workspace/target/release/dynamo-run /workspace/deploy/sdk/src/dynamo/sdk/cli/bin/dynamo-run

RUN cd /workspace/lib/bindings/python && \
    uv build --wheel --out-dir /workspace/dist --python 3.12
RUN cd /workspace && \
    uv build --wheel --out-dir /workspace/dist

# Runtime image
FROM dynamo-base as runtime
WORKDIR /workspace
COPY container/deps/requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

# Install built wheels
COPY --from=builder /workspace/dist/*.whl /tmp/wheels/
RUN uv pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels
