# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Base toolchain for Dynamo Cloud Agent development (no CUDA / engine wheels).

FROM ubuntu:24.04

ARG PROTOC_VERSION=30.2

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        libstdc++-13-dev \
        ca-certificates \
        cmake \
        curl \
        git \
        etcd-server \
        libclang-dev \
        libhwloc-dev \
        libudev-dev \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        sudo \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
        -o /tmp/protoc.zip \
    && unzip -o /tmp/protoc.zip -d /usr/local bin/protoc include/* \
    && rm /tmp/protoc.zip \
    && test -f /usr/local/include/google/protobuf/struct.proto \
    && protoc --version \
    && curl -fsSL "https://github.com/nats-io/nats-server/releases/download/v2.14.8/nats-server-v2.14.8-linux-amd64.tar.gz" \
        | tar -xzf - -C /usr/local/bin --strip-components=1 nats-server-v2.14.8-linux-amd64/nats-server

# Rust 1.96.1 (matches rust-toolchain.toml)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.96.1 --profile minimal \
    && ln -sf /root/.cargo/bin/* /usr/local/bin/

# uv on default PATH (login shells do not source ~/.local/bin)
RUN curl -fsSL https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

RUN gcc_ver="$(gcc -dumpversion | cut -d. -f1)" \
    && ln -sf "/usr/lib/gcc/x86_64-linux-gnu/${gcc_ver}/libstdc++.so" /usr/lib/x86_64-linux-gnu/libstdc++.so

RUN useradd -m -s /bin/bash ubuntu 2>/dev/null || true

USER ubuntu
WORKDIR /home/ubuntu
