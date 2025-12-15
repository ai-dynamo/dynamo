# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-architecture Dockerfile for Dynamo Operator
# Supports both amd64 and arm64 architectures

# Base stage - Common setup for all stages
ARG DOCKER_PROXY
ARG BASE_IMAGE="golang"
ARG BASE_IMAGE_TAG="1.24"
ARG RUNTIME_IMAGE="nvcr.io/nvidia/distroless/go"
ARG RUNTIME_IMAGE_TAG="v3.1.13"

FROM ${DOCKER_PROXY}${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Docker buildx automatically provides these
ARG TARGETOS=linux
ARG TARGETARCH

RUN echo "Building for ${TARGETOS}/${TARGETARCH}"

# Install common dependencies
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy go mod and sum files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Lint stage
FROM base AS linter

# Install golangci-lint using go install (builds with current Go version)
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.62.2

# Run linting
RUN golangci-lint run --timeout=5m

# Test stage
FROM base AS tester

# Install make if not present
RUN apt-get update && apt-get install -y make && apt-get clean && rm -rf /var/lib/apt/lists/*

# Run tests using Makefile
RUN make test

# Build stage - depends on successful lint and test
FROM base AS builder

ARG TARGETOS=linux
ARG TARGETARCH

# Build the binary for the target architecture
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o manager ./cmd/main.go

#########################################################################
#  Branch & Merge Pattern to handle architecture naming
#    This maps TARGETARCH (amd64/arm64) to architecture-specific settings
#########################################################################

##################################
########## Runtime Base ##########
##################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime_base
WORKDIR /

##################################
# Branch 1: AMD64 specific settings
##################################

FROM runtime_base AS runtime_amd64
# AMD64-specific configurations can be added here if needed

##################################
# Branch 2: ARM64 specific settings
##################################

FROM runtime_base AS runtime_arm64
# ARM64-specific configurations can be added here if needed

##################################
# Merge: Select correct branch
##################################

ARG TARGETARCH
FROM runtime_${TARGETARCH} AS runtime


COPY --from=builder /workspace/manager .
USER 65532:65532

ENTRYPOINT ["./manager"]
