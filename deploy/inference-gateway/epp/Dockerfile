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

ARG RUST_IMAGE=rust:1.93.1
ARG BUILDER_IMAGE=golang:1.25
ARG BASE_IMAGE=ubuntu:24.04

# =============================================================================
# Stage 1: Build Dynamo FFI static library (Rust)
# =============================================================================
FROM ${RUST_IMAGE} AS rust-builder

# TARGETARCH is provided automatically by buildx for multi-platform builds.
ARG TARGETARCH

# sccache configuration (content-addressed S3 cache keyed by source hash).
# When USE_SCCACHE=true, the build authenticates to S3 via IRSA secrets
# mounted on the cargo-build RUN step. If the sccache server fails to start
# (e.g. missing creds), the build continues without cache -- never with a
# stale one, unlike a persistent target/ mount.
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# etcd-client crate requires protoc to compile proto files
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /dynamo

# Install sccache using the shared helper so this build matches the rest
# of the repo's Rust builds (see container/templates/wheel_builder.Dockerfile).
COPY --from=dynamo container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}

# Copy Cargo workspace manifests, lockfile, and README (some crates inherit
# readme.workspace = true, so cargo needs README.md at the workspace root)
COPY --from=dynamo .cargo/ .cargo/
COPY --from=dynamo Cargo.toml Cargo.lock README.md ./

# Copy all workspace crates (libdynamo_llm depends transitively on many)
COPY --from=dynamo lib/ lib/

# Build libdynamo_llm with sccache for cross-run caching. The registry and
# git caches are content-addressed (safe to persist); no target/ mount --
# sccache caches compilations in S3 where stale artifacts can't be linked
# against newer source.
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETARCH},sharing=shared \
    --mount=type=cache,target=/usr/local/cargo/git,id=cargo-git-${TARGETARCH},sharing=shared \
    --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-epp-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cargo build --release -p libdynamo_llm && \
    mkdir -p /out && \
    cp target/release/libdynamo_llm_capi.a /out/ && \
    HEADER=$(find target/release/build -name llm_engine.h -path "*/out/*" | head -1) && \
    [ -n "$HEADER" ] && cp "$HEADER" /out/ || { echo "ERROR: llm_engine.h not found in target/"; exit 1; } && \
    if [ "$USE_SCCACHE" = "true" ]; then /tmp/use-sccache.sh show-stats "libdynamo_llm" || true; fi

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

# Install build dependencies for CGO + git (needed by cyclonedx-gomod's
# `git describe` version detection — see SBOM stage below).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy go mod files first for better caching (from default context = epp/)
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    go mod download

# Copy Go source (from default context = epp/)
COPY . .

# Place Rust artifacts where CGO expects them (see plugin.go #cgo directives)
COPY --from=rust-builder /out/libdynamo_llm_capi.a pkg/plugins/dynamo_kv_scorer/lib/
COPY --from=rust-builder /out/llm_engine.h pkg/plugins/dynamo_kv_scorer/include/

RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    --mount=type=cache,target=/root/.cache/go-build,sharing=shared \
    CGO_ENABLED=1 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -ldflags="-X sigs.k8s.io/gateway-api-inference-extension/version.GitVersion=${BUILD_REF} \
              -X sigs.k8s.io/gateway-api-inference-extension/version.GitCommit=${COMMIT_SHA}" \
    -o epp ./cmd/epp

# Compliance: emit a CycloneDX SBOM for EPP's Go module set so the frontend
# build (which COPYs /epp from this image) can consume it in its licenses
# stage. EPP itself is not customer-facing — the SBOM lives here purely as
# a build artifact for frontend's compliance pipeline. cyclonedx-gomod is
# pinned to v1.7.0 to keep the SBOM tool itself reproducible.
#
# cyclonedx-gomod app runs `git describe` to determine the module's version.
# The COPY of the build context doesn't bring .git/ (BuildKit excludes it
# unless an .dockerignore opt-in is set), so the tool fails with "no git
# repository found". Init a throwaway repo with a single tagged commit so
# `git describe` succeeds. BUILD_REF is the dynamo commit SHA the rest of
# this Dockerfile already stamps into the binary; reusing it keeps the
# SBOM's module version consistent with the binary's GitCommit string.
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    --mount=type=cache,target=/root/.cache/go-build,sharing=shared \
    git init -q . && \
    git -c user.email=ci@example -c user.name=ci add -A && \
    git -c user.email=ci@example -c user.name=ci commit -q --allow-empty -m "sbom build" && \
    git tag "v0.0.0-${BUILD_REF:-dev}" && \
    GOBIN=/usr/local/bin go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@v1.7.0 && \
    cyclonedx-gomod app -licenses -json \
        -main ./cmd/epp \
        -output /sbom-go.cdx.json \
        .

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

# EPP is not shipped to customers, but the frontend image's licenses stage
# consumes this CycloneDX SBOM via COPY --from=epp /sbom-go.cdx.json.
COPY --from=go-builder /sbom-go.cdx.json /sbom-go.cdx.json

# Note: EPP config is mounted via Kubernetes ConfigMap at runtime
# See helm/dynamo-gaie/templates/epp-configmap.yaml

RUN useradd -r -u 65532 -g nogroup nonroot
USER 65532:65534

ENTRYPOINT ["/epp"]
