# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified Dockerfile for snapshot-agent and placeholder images.
#
# Build targets (the `operator` build context is required — the snapshot module depends on the
# operator module via `replace => ../operator`):
#   docker build --platform linux/amd64 --target agent --build-context operator=../operator -t snapshot-agent:latest .
#   docker build --platform linux/amd64 --target placeholder --build-context operator=../operator --build-arg BASE_IMAGE=<app-image> -t placeholder:latest .
#
# Optional targets for CI:
#   docker build --target linter --build-context operator=../operator .   # Run linting
#   docker build --target tester --build-context operator=../operator .   # Run tests

# =============================================================================
# Build Arguments
# =============================================================================
ARG DOCKER_PROXY
ARG GO_VERSION=1.26.3
ARG CRIU_REPO=https://github.com/galletas1712/criu.git
ARG CRIU_REF=3dd2420c16e54fb7cd08b3e511d3a8bfb03cd001
ARG CRIU_SHA=3dd2420c16e54fb7cd08b3e511d3a8bfb03cd001
ARG CUDA_CHECKPOINT_REPO=https://github.com/NVIDIA/cuda-checkpoint.git
ARG CUDA_CHECKPOINT_REF=00d5cce84c628088d6caa203fc4af40c1538b6f7
ARG CUDA_CHECKPOINT_SHA=00d5cce84c628088d6caa203fc4af40c1538b6f7
ARG AGENT_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04

# For placeholder target only - this default allows agent builds to succeed,
# but placeholder builds MUST override it with --build-arg BASE_IMAGE=<image>
ARG BASE_IMAGE=placeholder-requires-base-image-arg
ARG BASE_IMAGE_USER=root

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

# The snapshot module depends on the operator module via `replace => ../operator`, so the operator
# module must sit at /operator (a sibling of /workspace) for `go mod download` and the Go build to
# resolve it. Provided via the `operator` named build context (see build-deploy-component action).
COPY --from=operator . /operator/

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
# Build against the oldest supported placeholder libc; the resulting binaries
# also run in the newer snapshot-agent image.
FROM ubuntu:22.04 AS criu-builder

ARG CRIU_REPO
ARG CRIU_REF
ARG CRIU_SHA
ARG CUDA_CHECKPOINT_REPO
ARG CUDA_CHECKPOINT_REF
ARG CUDA_CHECKPOINT_SHA

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
    python3-pip \
    python3-protobuf \
    libgnutls28-dev \
    libnftables-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init /tmp/criu \
    && cd /tmp/criu \
    && git remote add origin "${CRIU_REPO}" \
    && if [ "${CRIU_REF}" != "${CRIU_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CRIU_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CRIU_SHA}"; \
       fi \
    && git fetch --depth 1 origin "${CRIU_SHA}" \
    && git checkout --detach "${CRIU_SHA}" \
    && test "$(git rev-parse HEAD)" = "${CRIU_SHA}" \
    && make -j$(nproc) \
    && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin

COPY plugins/inet-remap/snapshot_inet_remap.c /tmp/snapshot_inet_remap.c
RUN mkdir -p /criu-install/usr/local/lib/snapshot/criu-plugins \
    && gcc -O2 -Wall -Wextra -fPIC -shared \
      -idirafter /tmp/criu/criu/include \
      -o /criu-install/usr/local/lib/snapshot/criu-plugins/snapshot_inet_remap.so \
      /tmp/snapshot_inet_remap.c

RUN git init /tmp/cuda-checkpoint \
    && cd /tmp/cuda-checkpoint \
    && git remote add origin "${CUDA_CHECKPOINT_REPO}" \
    && if [ "${CUDA_CHECKPOINT_REF}" != "${CUDA_CHECKPOINT_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CUDA_CHECKPOINT_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CUDA_CHECKPOINT_SHA}"; \
       fi \
    && git fetch --depth 1 origin "${CUDA_CHECKPOINT_SHA}" \
    && git checkout --detach "${CUDA_CHECKPOINT_SHA}" \
    && test "$(git rev-parse HEAD)" = "${CUDA_CHECKPOINT_SHA}"

RUN printf '%s\n' \
      "criu_source_url=${CRIU_REPO}" \
      "criu_source_ref=${CRIU_REF}" \
      "criu_source_sha=${CRIU_SHA}" \
      "cuda_checkpoint_source_url=${CUDA_CHECKPOINT_REPO}" \
      "cuda_checkpoint_source_ref=${CUDA_CHECKPOINT_REF}" \
      "cuda_checkpoint_source_sha=${CUDA_CHECKPOINT_SHA}" \
      > /restore-tool-source-provenance.txt

# =============================================================================
# Stage: Agent - Final snapshot-agent image
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS agent

ARG TARGETARCH=amd64

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64 (cuda-checkpoint has no ${TARGETARCH} binary)" >&2; exit 1; \
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
COPY --from=criu-builder /restore-tool-source-provenance.txt /opt/dynamo/restore-tool-source-provenance.txt
RUN criu --version

# Copy CUDA checkpoint binaries
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=criu-builder /tmp/cuda-checkpoint/LICENSE /legal/cuda-checkpoint/LICENSE
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
RUN chmod +x /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint-helper

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
# Dynamo backend entrypoints run a restore standby path when the operator
# injects DYN_SNAPSHOT_RESTORE_STANDBY=1, then sleep while the
# snapshot-agent restores into the container.
# =============================================================================
FROM ${BASE_IMAGE} AS placeholder

ARG BASE_IMAGE
ARG BASE_IMAGE_USER
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64 (cuda-checkpoint has no ${TARGETARCH} binary)" >&2; exit 1; \
    fi

# Install minimal runtime dependencies for CRIU restore (nsrestore runs here via nsenter)
RUN apt-get update && \
    gnutls_package="$(if apt-cache show libgnutls30t64 >/dev/null 2>&1; then \
        echo libgnutls30t64; \
    else \
        echo libgnutls30; \
    fi)" && \
    apt-get install -y --no-install-recommends \
    libbsd0 \
    libcap2 \
    libnet1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libprotobuf-c1 \
    "${gnutls_package}" \
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
COPY --from=criu-builder /restore-tool-source-provenance.txt /opt/dynamo/restore-tool-source-provenance.txt
RUN criu --version && echo "CRIU installed successfully"

# Copy CUDA checkpoint binaries
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=criu-builder /tmp/cuda-checkpoint/LICENSE /legal/cuda-checkpoint/LICENSE
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
RUN chmod +x /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint-helper

# Copy nsrestore binary (invoked by DaemonSet via nsenter)
COPY --from=builder /nsrestore /usr/local/bin/nsrestore
RUN chmod +x /usr/local/bin/nsrestore

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /var/criu-work

USER ${BASE_IMAGE_USER}
