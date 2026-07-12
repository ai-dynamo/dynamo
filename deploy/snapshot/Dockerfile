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
ARG NIXL_REPO=https://github.com/ai-dynamo/nixl.git
# Immutable pin containing "fix(posix): surface and recover from short writes
# in all io queues". Do not replace with v1.1.0 or a mutable PR/branch ref.
ARG NIXL_REF=71ca3cb249a481ae892fcc042a456e1a9b87463d
ARG AGENT_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04

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
# Stage: NIXL SDK/runtime builder (POSIX backend only)
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS nixl-builder

ARG NIXL_REPO
ARG NIXL_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libaio-dev \
    meson \
    ninja-build \
    pkg-config \
    pybind11-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Build a test configuration first. Keep its generic warm-up below Docker's
# 1024-FD limit; the upstream short-write/error-recovery matrix is independent
# of this count. Package a separate release configuration afterward.
RUN git init /tmp/nixl \
    && cd /tmp/nixl \
    && git remote add origin ${NIXL_REPO} \
    && git fetch --depth 1 origin ${NIXL_REF} \
    && test "$(git rev-parse FETCH_HEAD)" = "${NIXL_REF}" \
    && git checkout --detach ${NIXL_REF} \
    && CXXFLAGS="-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized" meson setup build-test \
      --buildtype=debugoptimized \
      -Dbuild_tests=true \
      -Dbuild_examples=false \
      -Denable_plugins=POSIX \
    && meson compile -C build-test \
    && NIXL_PLUGIN_DIR=/tmp/nixl/build-test/src/plugins/posix \
      ./build-test/test/unit/plugins/posix/nixl_posix_test -n 32 \
    && meson setup build \
      --prefix=/usr/local \
      --libdir=lib \
      --buildtype=release \
      -Dbuild_tests=false \
      -Dbuild_examples=false \
      -Denable_plugins=POSIX \
    && meson compile -C build \
    && DESTDIR=/nixl-install meson install -C build --strip \
    && mkdir -p /nixl-runtime/lib/plugins \
    && printf '%s\n' "${NIXL_REF}" > /nixl-runtime/NIXL_COMMIT \
    && cp /tmp/nixl/subprojects/liburing-liburing-2.14/LICENSE /nixl-runtime/liburing-LICENSE \
    && cp \
      /nixl-install/usr/local/lib/libnixl.so \
      /nixl-install/usr/local/lib/libnixl_build.so \
      /nixl-install/usr/local/lib/libnixl_common.so \
      /nixl-install/usr/local/lib/libserdes.so \
      /nixl-install/usr/local/lib/libstream.so \
      /nixl-install/usr/local/lib/libfile_utils.so \
      /nixl-runtime/lib/ \
    && cp -a /nixl-install/usr/local/lib/liburing.so* /nixl-runtime/lib/ \
    && cp /nixl-install/usr/local/lib/plugins/libplugin_POSIX.so /nixl-runtime/lib/plugins/

# =============================================================================
# Stage: CUDA checkpoint helper builder
# =============================================================================
FROM ${AGENT_BASE_IMAGE} AS cuda-helper-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=nixl-builder /nixl-install/usr/local /usr/local
COPY cmd/cuda-checkpoint-helper/main.cpp ./cmd/cuda-checkpoint-helper/main.cpp
COPY cmd/cuda-checkpoint-helper/cuda_checkpoint_compat.h ./cmd/cuda-checkpoint-helper/cuda_checkpoint_compat.h
COPY cmd/cuda-checkpoint-helper/daemon_protocol.cpp ./cmd/cuda-checkpoint-helper/daemon_protocol.cpp
COPY cmd/cuda-checkpoint-helper/daemon_protocol.h ./cmd/cuda-checkpoint-helper/daemon_protocol.h
COPY cmd/cuda-checkpoint-helper/daemon_protocol_test.cpp ./cmd/cuda-checkpoint-helper/daemon_protocol_test.cpp
COPY cmd/cuda-checkpoint-helper/storage_manifest.cpp ./cmd/cuda-checkpoint-helper/storage_manifest.cpp
COPY cmd/cuda-checkpoint-helper/storage_manifest.h ./cmd/cuda-checkpoint-helper/storage_manifest.h
COPY cmd/cuda-checkpoint-helper/storage_manifest_test.cpp ./cmd/cuda-checkpoint-helper/storage_manifest_test.cpp
COPY cmd/cuda-checkpoint-helper/transfer_config.cpp ./cmd/cuda-checkpoint-helper/transfer_config.cpp
COPY cmd/cuda-checkpoint-helper/transfer_config.h ./cmd/cuda-checkpoint-helper/transfer_config.h
COPY cmd/cuda-checkpoint-helper/transfer_config_test.cpp ./cmd/cuda-checkpoint-helper/transfer_config_test.cpp
COPY cmd/cuda-checkpoint-helper/transfer_engine.cpp ./cmd/cuda-checkpoint-helper/transfer_engine.cpp
COPY cmd/cuda-checkpoint-helper/transfer_engine.h ./cmd/cuda-checkpoint-helper/transfer_engine.h
COPY cmd/cuda-checkpoint-helper/benchmark.cpp ./cmd/cuda-checkpoint-helper/benchmark.cpp
COPY cmd/cuda-checkpoint-helper/benchmark_cache.cpp ./cmd/cuda-checkpoint-helper/benchmark_cache.cpp
COPY cmd/cuda-checkpoint-helper/benchmark_cache.h ./cmd/cuda-checkpoint-helper/benchmark_cache.h
COPY cmd/cuda-checkpoint-helper/benchmark_cache_test.cpp ./cmd/cuda-checkpoint-helper/benchmark_cache_test.cpp
COPY cmd/cuda-checkpoint-helper/benchmark_storage.cpp ./cmd/cuda-checkpoint-helper/benchmark_storage.cpp
COPY cmd/cuda-checkpoint-helper/benchmark_storage.h ./cmd/cuda-checkpoint-helper/benchmark_storage.h
COPY cmd/cuda-checkpoint-helper/benchmark_storage_test.cpp ./cmd/cuda-checkpoint-helper/benchmark_storage_test.cpp

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -o /cuda-checkpoint-helper \
    ./cmd/cuda-checkpoint-helper/main.cpp \
    ./cmd/cuda-checkpoint-helper/daemon_protocol.cpp \
    ./cmd/cuda-checkpoint-helper/storage_manifest.cpp \
    ./cmd/cuda-checkpoint-helper/transfer_config.cpp \
    ./cmd/cuda-checkpoint-helper/transfer_engine.cpp \
    -I/usr/local/cuda/include \
    -I/usr/local/include \
    -L/usr/local/cuda/lib64/stubs \
    -L/usr/local/lib \
    -Wl,-rpath,/usr/local/lib \
    -lnixl -lnixl_build -lcuda -ldl

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread \
      -o /cuda-checkpoint-helper-daemon-protocol-test \
      ./cmd/cuda-checkpoint-helper/daemon_protocol.cpp \
      ./cmd/cuda-checkpoint-helper/daemon_protocol_test.cpp \
    && /cuda-checkpoint-helper-daemon-protocol-test

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -o /cuda-nixl-posix-benchmark \
    ./cmd/cuda-checkpoint-helper/benchmark.cpp \
    ./cmd/cuda-checkpoint-helper/benchmark_cache.cpp \
    ./cmd/cuda-checkpoint-helper/benchmark_storage.cpp \
    ./cmd/cuda-checkpoint-helper/transfer_config.cpp \
    ./cmd/cuda-checkpoint-helper/transfer_engine.cpp \
    -I/usr/local/cuda/include \
    -I/usr/local/include \
    -L/usr/local/cuda/lib64/stubs \
    -L/usr/local/lib \
    -Wl,-rpath,/usr/local/lib \
    -lnixl -lnixl_build -lcuda

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -o /cuda-checkpoint-helper-storage-test \
      ./cmd/cuda-checkpoint-helper/storage_manifest.cpp \
      ./cmd/cuda-checkpoint-helper/storage_manifest_test.cpp \
    && /cuda-checkpoint-helper-storage-test

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -o /cuda-checkpoint-helper-transfer-test \
      ./cmd/cuda-checkpoint-helper/transfer_config.cpp \
      ./cmd/cuda-checkpoint-helper/transfer_config_test.cpp \
    && /cuda-checkpoint-helper-transfer-test

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -o /cuda-checkpoint-helper-benchmark-cache-test \
      ./cmd/cuda-checkpoint-helper/benchmark_cache.cpp \
      ./cmd/cuda-checkpoint-helper/benchmark_cache_test.cpp \
    && /cuda-checkpoint-helper-benchmark-cache-test

RUN g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -o /cuda-checkpoint-helper-benchmark-storage-test \
      ./cmd/cuda-checkpoint-helper/benchmark_storage.cpp \
      ./cmd/cuda-checkpoint-helper/transfer_config.cpp \
      ./cmd/cuda-checkpoint-helper/benchmark_storage_test.cpp \
    && /cuda-checkpoint-helper-benchmark-storage-test

# =============================================================================
# Stage: CRIU Builder - Build CRIU with CUDA plugin
# =============================================================================
FROM ubuntu:24.04 AS criu-builder

ARG CRIU_REPO
ARG CRIU_REF

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

COPY plugins/inet-remap/snapshot_inet_remap.c /tmp/snapshot_inet_remap.c
RUN mkdir -p /criu-install/usr/local/lib/snapshot/criu-plugins \
    && gcc -O2 -Wall -Wextra -fPIC -shared \
      -idirafter /tmp/criu/criu/include \
      -o /criu-install/usr/local/lib/snapshot/criu-plugins/snapshot_inet_remap.so \
      /tmp/snapshot_inet_remap.c

RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint

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
    libaio1t64 \
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

# Copy CUDA checkpoint binaries
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=criu-builder /tmp/cuda-checkpoint/LICENSE /legal/cuda-checkpoint/LICENSE
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
COPY --from=cuda-helper-builder /cuda-nixl-posix-benchmark /usr/local/bin/cuda-nixl-posix-benchmark
COPY --from=nixl-builder /nixl-runtime/lib /usr/local/lib
COPY --from=nixl-builder /nixl-runtime/NIXL_COMMIT /usr/local/share/nixl/NIXL_COMMIT
COPY --from=nixl-builder /tmp/nixl/LICENSE /legal/NIXL/LICENSE
COPY --from=nixl-builder /nixl-runtime/liburing-LICENSE /legal/liburing/LICENSE
ENV NIXL_PLUGIN_DIR=/usr/local/lib/plugins
RUN ldconfig
RUN chmod +x \
    /usr/local/sbin/cuda-checkpoint \
    /usr/local/bin/cuda-checkpoint-helper \
    /usr/local/bin/cuda-nixl-posix-benchmark

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
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64 (cuda-checkpoint has no ${TARGETARCH} binary)" >&2; exit 1; \
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
    libaio1t64 \
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

# Copy CUDA checkpoint binaries
COPY --from=criu-builder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=criu-builder /tmp/cuda-checkpoint/LICENSE /legal/cuda-checkpoint/LICENSE
COPY --from=cuda-helper-builder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
COPY --from=nixl-builder /nixl-runtime/lib /usr/local/lib
COPY --from=nixl-builder /nixl-runtime/NIXL_COMMIT /usr/local/share/nixl/NIXL_COMMIT
COPY --from=nixl-builder /tmp/nixl/LICENSE /legal/NIXL/LICENSE
COPY --from=nixl-builder /nixl-runtime/liburing-LICENSE /legal/liburing/LICENSE
ENV NIXL_PLUGIN_DIR=/usr/local/lib/plugins
RUN ldconfig
RUN chmod +x /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint-helper

# Copy nsrestore binary (invoked by DaemonSet via nsenter)
COPY --from=builder /nsrestore /usr/local/bin/nsrestore
RUN chmod +x /usr/local/bin/nsrestore

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /var/criu-work
