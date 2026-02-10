# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified Dockerfile for chrek-agent and placeholder images.
#
# Build targets:
#   docker build --target agent -t chrek-agent:latest .
#   docker build --target placeholder --build-arg BASE_IMAGE=<app-image> -t placeholder:latest .
#
# Optional targets for CI:
#   docker build --target linter .   # Run linting
#   docker build --target tester .   # Run tests

# =============================================================================
# Build Arguments
# =============================================================================
ARG DOCKER_PROXY
ARG GO_VERSION=1.25
ARG CRIU_VERSION=v4.2
ARG AGENT_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04

# For placeholder target only - this default allows agent builds to succeed,
# but placeholder builds MUST override it with --build-arg BASE_IMAGE=<image>
ARG BASE_IMAGE=placeholder-requires-base-image-arg

# =============================================================================
# Stage: Go base - Common setup for Go builds
# =============================================================================
FROM ${DOCKER_PROXY}golang:${GO_VERSION} AS go-base

ARG TARGETOS=linux
ARG TARGETARCH=amd64

RUN echo "Building for ${TARGETOS}/${TARGETARCH}"

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# =============================================================================
# Stage: Linter - Run golangci-lint
# =============================================================================
FROM go-base AS linter

RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.62.2
RUN golangci-lint run --timeout=5m

# =============================================================================
# Stage: Tester - Run tests
# =============================================================================
FROM go-base AS tester

RUN go test ./... -v

# =============================================================================
# Stage: Builder - Build Go binaries
# =============================================================================
FROM go-base AS builder

ARG TARGETOS=linux
ARG TARGETARCH=amd64

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /chrek-agent ./cmd/agent
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /restore-entrypoint ./cmd/restore-entrypoint

# =============================================================================
# Stage: CRIU Builder - Build CRIU with CUDA plugin
# =============================================================================
FROM ubuntu:24.04 AS criu-builder

ARG CRIU_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    pkg-config \
    libbsd-dev \
    libcap-dev \
    libnet1-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    libprotobuf-dev \
    libprotobuf-c-dev \
    protobuf-c-compiler \
    protobuf-compiler \
    python3 \
    python3-protobuf \
    libgnutls28-dev \
    libnftables-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch ${CRIU_VERSION} https://github.com/checkpoint-restore/criu.git /tmp/criu \
    && cd /tmp/criu \
    && make -j$(nproc) \
    && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin

RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint

# =============================================================================
# Stage: Agent - Final chrek-agent image
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS agent

# Install CRIU runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libbsd0 \
    libcap2 \
    libnet1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libprotobuf-c1 \
    libgnutls30t64 \
    libnftables1 \
    iproute2 \
    iptables \
    procps \
    uuid-runtime \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy CRIU from builder
COPY --from=criu-builder /criu-install/usr/local /usr/local
RUN criu --version

# Copy cuda-checkpoint binary
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
RUN chmod +x /usr/local/sbin/cuda-checkpoint

# Copy the built binaries
COPY --from=builder /chrek-agent /usr/local/bin/chrek-agent
COPY --from=builder /restore-entrypoint /restore-entrypoint

# Create checkpoint directory
RUN mkdir -p /checkpoints

# Set environment variables
ENV HOST_PROC=/host/proc \
    CONTAINERD_SOCKET=/run/containerd/containerd.sock \
    CHECKPOINT_DIR=/checkpoints \
    LISTEN_ADDR=:8080

EXPOSE 8080

USER root

ENTRYPOINT ["/usr/local/bin/chrek-agent"]

# =============================================================================
# Stage: Placeholder - Restore placeholder image (requires BASE_IMAGE arg)
# =============================================================================
FROM ${BASE_IMAGE} AS placeholder

ARG BASE_IMAGE
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

# Install CRIU runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libbsd0 \
    libcap2 \
    libnet1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libprotobuf-c1 \
    libgnutls30 \
    libnftables1 \
    iproute2 \
    iptables \
    procps \
    uuid-runtime \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy CRIU from builder
COPY --from=criu-builder /criu-install/usr/local /usr/local
RUN criu --version && echo "CRIU installed successfully"

# Copy cuda-checkpoint binary
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
RUN chmod +x /usr/local/sbin/cuda-checkpoint

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /tmp /var/criu-work

# Copy restore binaries
COPY --from=builder /restore-entrypoint /restore-entrypoint
RUN chmod +x /restore-entrypoint

COPY scripts/smart-entrypoint.sh /smart-entrypoint.sh
RUN chmod +x /smart-entrypoint.sh

# Set environment variables
ENV DYN_CHECKPOINT_PATH=/checkpoints \
    RESTORE_TRIGGER=/tmp/restore-trigger \
    RESTORE_WAIT_TIMEOUT=300 \
    CRIU_LOG_LEVEL=4 \
    WAIT_FOR_CHECKPOINT=0 \
    CUDA_PLUGIN_DIR=/usr/local/lib/criu \
    DEBUG=0

ENTRYPOINT ["/smart-entrypoint.sh"]
CMD []
