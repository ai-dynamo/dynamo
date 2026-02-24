{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dynamo_base.Dockerfile ===
##################################
########## Base Image ############
##################################

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS dynamo_base

ARG ARCH
ARG ARCH_ALT

USER root
WORKDIR /opt/dynamo

# Install sccache into the base image so downstream stages can COPY it
# instead of downloading from GitHub (avoids 502 errors under parallel builds)
ARG SCCACHE_VERSION=v0.14.0
RUN wget --tries=3 --waitretry=5 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    tar -xzf "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    mv "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl/sccache" /usr/local/bin/ && \
    rm -rf sccache*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install NATS server
ARG NATS_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    wget --tries=3 --waitretry=5 https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${ARCH}.deb && \
    dpkg -i nats-server-${NATS_VERSION}-${ARCH}.deb && rm nats-server-${NATS_VERSION}-${ARCH}.deb

# Install etcd
ARG ETCD_VERSION
RUN wget --tries=3 --waitretry=5 https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-${ARCH}.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=/usr/local/bin/etcd/:$PATH

# Rust Setup
# Rust environment setup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.90.0

# Define Rust target based on ARCH_ALT ARG
ARG RUSTARCH=${ARCH_ALT}-unknown-linux-gnu

# Install Rust
RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME
