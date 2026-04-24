# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build from the repo root so all COPY paths resolve:
#   docker build -f recipes/nemotron-omni/Dockerfile -t nemotron-omni .

ARG BASE_IMAGE="vllm/vllm-openai:cu130-nightly-8936118134d0547fa1cc78adab2d03edd6d3dc48"
FROM ${BASE_IMAGE}

USER root
WORKDIR /workspace

# System build dependencies.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        git \
        libclang-dev \
        libhwloc-dev \
        libudev-dev \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain for maturin and dynamo._core.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    RUST_VERSION=1.93.1

RUN ARCH_ALT=$(uname -m) && \
    RUSTARCH="${ARCH_ALT}-unknown-linux-gnu" && \
    wget -q --tries=3 --waitretry=5 \
        "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal \
        --default-toolchain ${RUST_VERSION} --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w ${RUSTUP_HOME} ${CARGO_HOME}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# NATS server and etcd for the Dynamo service mesh.
ARG NATS_VERSION=v2.10.28
ARG ETCD_VERSION=v3.5.21

RUN ARCH=$(dpkg --print-architecture) && \
    wget -q --tries=3 --waitretry=5 \
        "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${ARCH}.deb" && \
    dpkg -i nats-server-${NATS_VERSION}-${ARCH}.deb && \
    rm nats-server-${NATS_VERSION}-${ARCH}.deb && \
    wget -q --tries=3 --waitretry=5 \
        "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${ARCH}.tar.gz" \
        -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    DYNAMO_HOME=/workspace \
    CARGO_TARGET_DIR=/workspace/.cargo-target \
    CUDA_HOME=/usr/local/cuda \
    CPATH=/usr/local/cuda/include \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility \
    PATH=/opt/dynamo/venv/bin:/usr/local/cargo/bin:/usr/local/bin/etcd:/usr/local/cuda/bin:${PATH}

# Inherit the base image's vLLM and torch installations.
RUN python3 -m venv --system-site-packages "${VIRTUAL_ENV}" && \
    uv pip install pip setuptools wheel "maturin[patchelf]"

COPY . /workspace/

# Build Dynamo from source. Install without dependencies to avoid replacing the
# CUDA-aware vLLM and torch packages from the base image.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/workspace/.cargo-target \
    cd /workspace/lib/bindings/python && \
    maturin develop --uv --release && \
    cd /workspace && \
    (uv pip install -e lib/gpu_memory_service 2>/dev/null || true) && \
    uv pip install --no-deps -e .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        librosa \
        "nixl[cu12]<=0.10.1" \
        --requirement /workspace/container/deps/requirements.common.txt \
        --requirement /workspace/container/deps/requirements.vllm.txt \
        --requirement /workspace/container/deps/requirements.planner.txt && \
    uv pip uninstall torch torchvision torchaudio nvidia-nccl-cu12 2>/dev/null; \
    echo "source /opt/dynamo/venv/bin/activate" >> /etc/bash.bashrc

# The upstream vLLM OpenAI image uses a serve entrypoint. Override it so this
# recipe behaves like a normal Dynamo runtime image.
ENTRYPOINT ["/bin/bash"]
