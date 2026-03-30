# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Base stage - Common setup for all stages
ARG DOCKER_PROXY
ARG BASE_IMAGE="golang"
ARG BASE_IMAGE_TAG="1.25"
FROM ${DOCKER_PROXY}${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Docker buildx automatically provides these
ARG TARGETOS=linux
ARG TARGETARCH

RUN echo "Building for ${TARGETOS}/${TARGETARCH}"

# Install common dependencies (python3-pip needed for generate-pydantic target)
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir pydantic --break-system-packages

WORKDIR /workspace/operator

# Copy module manifests first for better layer caching.
COPY operator/go.mod operator/go.sum ./
COPY snapshot /workspace/snapshot
RUN go mod download

# Copy operator source after dependencies are resolved.
COPY operator .

# Lint stage
FROM base AS linter

# Install golangci-lint using go install (builds with current Go version)
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.62.2

# Run linting
RUN golangci-lint run --timeout=5m

# Test stage
FROM base AS tester

# Install make if not present
RUN apt-get update && apt-get install -y --no-install-recommends make && apt-get clean && rm -rf /var/lib/apt/lists/*

# Run tests using Makefile
RUN make test

# Build stage - depends on successful lint and test
FROM base AS builder

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o manager ./cmd/main.go && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o crd-apply ./cmd/crd-apply/main.go

# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v4.0.3
WORKDIR /
COPY --from=builder /workspace/operator/manager .
COPY --from=builder /workspace/operator/crd-apply .
COPY --from=builder /workspace/operator/config/crd/bases/ /opt/dynamo-operator/crds/
USER 65532:65532

ENTRYPOINT ["./manager"]
