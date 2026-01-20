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
# Builds a custom EPP image with Dynamo KV-aware routing plugins
#
# PREREQUISITES: Run `make dynamo-lib` before building this image to ensure
# the Dynamo FFI library and headers are in place.

ARG BUILDER_IMAGE=golang:1.24-bookworm
ARG BASE_IMAGE=ubuntu:24.04

# =============================================================================
# Build stage
# =============================================================================
FROM ${BUILDER_IMAGE} AS builder

# Docker buildx provides these automatically for multi-platform builds
ARG TARGETOS=linux
ARG TARGETARCH

ARG COMMIT_SHA
ARG BUILD_REF

WORKDIR /workspace

# Install build dependencies for CGO
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the source code (including pre-built Dynamo library)
COPY . .

# Verify Dynamo library exists
RUN if [ ! -f "pkg/plugins/dynamo_kv_scorer/lib/libdynamo_llm_capi.a" ]; then \
        echo "ERROR: Dynamo library not found!"; \
        echo "Run 'make dynamo-lib' before building the Docker image."; \
        exit 1; \
    fi

# Build with CGO enabled for the Dynamo FFI
# Use TARGETOS/TARGETARCH from Docker buildx for proper platform support
RUN CGO_ENABLED=1 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -ldflags="-X sigs.k8s.io/gateway-api-inference-extension/version.GitVersion=${BUILD_REF} \
              -X sigs.k8s.io/gateway-api-inference-extension/version.GitCommit=${COMMIT_SHA}" \
    -o epp ./cmd/epp

# =============================================================================
# Runtime stage
# =============================================================================
FROM ${BASE_IMAGE}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy the binary from builder
COPY --from=builder /workspace/epp .

# Note: EPP config is mounted via Kubernetes ConfigMap at runtime
# See helm/dynamo-gaie/templates/epp-configmap.yaml

# Create non-root user
RUN useradd -r -u 65532 -g nogroup nonroot
USER 65532:65534

ENTRYPOINT ["/epp"]
