# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified Dockerfile for snapshot-agent and placeholder images.
#
# Build targets:
#   docker build --platform linux/amd64 --target agent -t snapshot-agent:latest .
#   docker build --platform linux/amd64 --target placeholder --build-arg BASE_IMAGE=<app-image> -t placeholder:latest .
#
# Optional targets for CI:
#   docker build --target linter .   # Run linting
#   docker build --target tester .   # Run tests
#
# For arm64 helper-only images, omit the amd64-only NVIDIA cuda-checkpoint
# launcher binary:
#   docker build --platform linux/arm64 --target agent \
#     --build-arg SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=false \
#     -t snapshot-agent:arm64 .

# =============================================================================
# Build Arguments
# =============================================================================
ARG DOCKER_PROXY
ARG GO_VERSION=1.26.3
# Default to upstream CRIU development branch. Custom forks can override both
# args at build time, for example:
#   --build-arg CRIU_REPO=<git-remote-url>
#   --build-arg CRIU_REF=<fork-branch-or-sha>
ARG CRIU_REPO=https://github.com/checkpoint-restore/criu.git
ARG CRIU_REF=criu-dev
ARG AGENT_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
ARG SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=true

# For placeholder target only - this default allows agent builds to succeed,
# but placeholder builds MUST override it with --build-arg BASE_IMAGE=<image>
ARG BASE_IMAGE=placeholder-requires-base-image-arg

# =============================================================================
# Stage: Go base - Common setup for Go builds
# =============================================================================
FROM ${DOCKER_PROXY}library/golang:${GO_VERSION} AS go-base

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

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /snapshot-agent ./cmd/agent
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /nsrestore ./cmd/nsrestore

# =============================================================================
# Stage: CUDA checkpoint helper builder
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS cuda-helper-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY cmd/cuda-checkpoint-helper/main.c ./cmd/cuda-checkpoint-helper/main.c

RUN gcc -O2 -Wall -Wextra -o /cuda-checkpoint-helper \
    ./cmd/cuda-checkpoint-helper/main.c \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64/stubs \
    -lcuda

# =============================================================================
# Stage: CRIU Builder - Build CRIU with CUDA plugin
# =============================================================================
FROM ubuntu:24.04 AS criu-builder

ARG CRIU_REPO
ARG CRIU_REF
ARG TARGETARCH=amd64
ARG SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=true

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

RUN git init /tmp/criu \
    && cd /tmp/criu \
    && git remote add origin ${CRIU_REPO} \
    && git fetch --depth 1 origin ${CRIU_REF} \
    && git checkout FETCH_HEAD \
    && make -j$(nproc) \
    && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin

RUN if [ "${SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY}" = "true" ]; then \
      if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "ERROR: cuda-checkpoint launcher binary is only available for amd64; set SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=false for helper-only ${TARGETARCH} builds" >&2; \
        exit 1; \
      fi; \
      git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint; \
    else \
      mkdir -p /tmp/cuda-checkpoint; \
    fi

# =============================================================================
# Stage: Optional CUDA checkpoint launcher binary installer
# =============================================================================
FROM criu-builder AS cuda-checkpoint-binary-installer

ARG SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=true

RUN mkdir -p /cuda-checkpoint-binary/usr/local/sbin /cuda-checkpoint-binary/legal/cuda-checkpoint \
    && if [ "${SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY}" = "true" ]; then \
      cp /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /cuda-checkpoint-binary/usr/local/sbin/cuda-checkpoint; \
      cp /tmp/cuda-checkpoint/LICENSE /cuda-checkpoint-binary/legal/cuda-checkpoint/LICENSE; \
    fi

# =============================================================================
# Stage: Agent - Final snapshot-agent image
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS agent

ARG TARGETARCH=amd64
ARG SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=true

RUN if [ "${SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY}" = "true" ] && [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: cuda-checkpoint launcher binary is only available for amd64; set SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=false for helper-only ${TARGETARCH} builds" >&2; exit 1; \
    fi

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
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Copy CRIU from builder
COPY --from=criu-builder /criu-install/usr/local /usr/local
COPY --from=criu-builder /tmp/criu/COPYING /legal/CRIU/COPYING
RUN criu --version

# Copy CUDA checkpoint tooling. cuda-checkpoint is optional for helper-only
# images; cuda-checkpoint-helper is always required by snapshot-agent.
COPY --from=cuda-checkpoint-binary-installer /cuda-checkpoint-binary/ /
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
RUN chmod +x /usr/local/bin/cuda-checkpoint-helper \
    && if [ -f /usr/local/sbin/cuda-checkpoint ]; then chmod +x /usr/local/sbin/cuda-checkpoint; fi

# Copy the built binaries
COPY --from=builder /snapshot-agent /usr/local/bin/snapshot-agent
COPY --from=builder /nsrestore /usr/local/bin/nsrestore

# Create directories
RUN mkdir -p /checkpoints /var/run/snapshot

USER root

ENTRYPOINT ["/usr/local/bin/snapshot-agent"]

# =============================================================================
# Stage: Placeholder - Runtime-compatible restore image (requires BASE_IMAGE arg)
# This image is a superset of the runtime image: same default execution contract
# (entrypoint/cmd/user), plus CRIU/cuda-checkpoint tooling for external restore.
# The operator may still override command to "sleep infinity" for restore pods.
# =============================================================================
FROM ${BASE_IMAGE} AS placeholder

ARG BASE_IMAGE
ARG TARGETARCH=amd64
ARG SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=true
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY}" = "true" ] && [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: cuda-checkpoint launcher binary is only available for amd64; set SNAPSHOT_USE_CUDA_CHECKPOINT_BINARY=false for helper-only ${TARGETARCH} builds" >&2; exit 1; \
    fi

# Install minimal runtime dependencies for CRIU restore (nsrestore runs here via nsenter)
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

# Copy CRIU from builder (needed by nsrestore running inside these namespaces)
COPY --from=criu-builder /criu-install/usr/local /usr/local
COPY --from=criu-builder /tmp/criu/COPYING /legal/CRIU/COPYING
RUN criu --version && echo "CRIU installed successfully"

# Copy CUDA checkpoint tooling. cuda-checkpoint is optional for helper-only
# images; cuda-checkpoint-helper is always required by nsrestore.
COPY --from=cuda-checkpoint-binary-installer /cuda-checkpoint-binary/ /
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
RUN chmod +x /usr/local/bin/cuda-checkpoint-helper \
    && if [ -f /usr/local/sbin/cuda-checkpoint ]; then chmod +x /usr/local/sbin/cuda-checkpoint; fi

# Copy nsrestore binary (invoked by DaemonSet via nsenter)
COPY --from=builder /nsrestore /usr/local/bin/nsrestore
RUN chmod +x /usr/local/bin/nsrestore

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /var/criu-work
