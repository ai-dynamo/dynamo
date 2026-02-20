# SPDX-FileCopyrightText:  Copyright The Kubernetes Authors.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES

# Dynamo EPP Dockerfile
# Self-contained 3-stage build: Rust FFI library → Go binary → Runtime
#
# Uses two build contexts (requires Docker BuildKit):
#   - Default context (.): the epp/ Go project
#   - Named context "dynamo": the Dynamo repository root (Rust workspace)
#
# Build via Makefile (recommended):
#   make image-load
#
# Build manually:
#   docker buildx build \
#     --build-context dynamo=../../.. \
#     -t dynamo/dynamo-epp:dev .

ARG RUST_IMAGE=rust:1.90
ARG BUILDER_IMAGE=golang:1.24
ARG BASE_IMAGE=ubuntu:24.04

# =============================================================================
# Stage 1: Build Dynamo FFI static library (Rust)
# =============================================================================
FROM ${RUST_IMAGE} AS rust-builder

# TARGETARCH is provided automatically by buildx for multi-platform builds.
# Used here only for per-platform cache isolation.
ARG TARGETARCH

# etcd-client crate requires protoc to compile proto files
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /dynamo

# Copy Cargo workspace manifests, lockfile, and README (some crates inherit
# readme.workspace = true, so cargo needs README.md at the workspace root)
COPY --from=dynamo Cargo.toml Cargo.lock README.md ./

# Copy all workspace crates (libdynamo_llm depends transitively on many)
COPY --from=dynamo lib/ lib/

# Build the static library; use BuildKit cache mounts for cargo registry and
# build artifacts so incremental Docker rebuilds are fast.
# Cache IDs are keyed by TARGETARCH so multi-platform builds don't collide.
# Artifacts are copied to /out/ because the target/ cache mount is ephemeral.
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETARCH} \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETARCH} \
    --mount=type=cache,target=/dynamo/target,id=cargo-target-${TARGETARCH} \
    cargo build --release -p libdynamo_llm && \
    mkdir -p /out && \
    cp target/release/libdynamo_llm_capi.a /out/ && \
    cp lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h /out/

# =============================================================================
# Stage 2: Build Go EPP binary
# =============================================================================
FROM ${BUILDER_IMAGE} AS go-builder

# Docker buildx provides these automatically for multi-platform builds
ARG TARGETOS=linux
ARG TARGETARCH

ARG COMMIT_SHA
ARG BUILD_REF

WORKDIR /workspace

# Install build dependencies for CGO
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy go mod files first for better caching (from default context = epp/)
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Copy Go source (from default context = epp/)
COPY . .

# Place Rust artifacts where CGO expects them (see plugin.go #cgo directives)
COPY --from=rust-builder /out/libdynamo_llm_capi.a pkg/plugins/dynamo_kv_scorer/lib/
COPY --from=rust-builder /out/llm_engine.h pkg/plugins/dynamo_kv_scorer/include/

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -ldflags="-X sigs.k8s.io/gateway-api-inference-extension/version.GitVersion=${BUILD_REF} \
              -X sigs.k8s.io/gateway-api-inference-extension/version.GitCommit=${COMMIT_SHA}" \
    -o epp ./cmd/epp

# =============================================================================
# Stage 3: Runtime
# =============================================================================
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY --from=go-builder /workspace/epp .

# Note: EPP config is mounted via Kubernetes ConfigMap at runtime
# See helm/dynamo-gaie/templates/epp-configmap.yaml

RUN useradd -r -u 65532 -g nogroup nonroot
USER 65532:65534

ENTRYPOINT ["/epp"]
