# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image of the Multi-Cluster Services registry: mcs-registry (the
# entrypoint), mcs-controller and mcs-admin, with the CRDs the controller
# needs in /opt/global-discovery-mcs/crds. Build it from this directory.

# Base stage - common setup for all stages
ARG DOCKER_PROXY
ARG BASE_IMAGE="library/golang"
ARG BASE_IMAGE_TAG="1.27.3"
FROM ${DOCKER_PROXY}${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Docker buildx automatically provides these
ARG TARGETOS=linux
ARG TARGETARCH

WORKDIR /workspace

# Copy go mod and sum files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Lint stage
FROM base AS linter

# Install golangci-lint using go install (builds with the current Go version)
RUN go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.15.0

# Run linting, integration tests included
RUN golangci-lint run --build-tags integration --timeout=10m

# Test stage: unit tests, the envtest suite and the coverage gate
FROM base AS tester

# Install make if not present
RUN apt-get update && apt-get install -y --no-install-recommends make && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN make test

# Build stage
FROM base AS builder

RUN for cmd in mcs-registry mcs-controller mcs-admin; do \
        CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
            go build -trimpath -ldflags="-s -w" -o /out/${cmd} ./cmd/${cmd} || exit 1; \
    done

# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v4.1.7
WORKDIR /
COPY --from=builder /out/mcs-registry /out/mcs-controller /out/mcs-admin /
COPY config/crd/bases /opt/global-discovery-mcs/crds/
USER 65532:65532

ENTRYPOINT ["/mcs-registry"]
