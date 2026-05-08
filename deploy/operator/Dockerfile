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

WORKDIR /workspace

# Copy go mod and sum files first for better layer caching
COPY go.mod go.sum ./
COPY --from=snapshot go.mod go.sum /snapshot/
RUN go mod download

# Copy source code
COPY . .
COPY --from=snapshot . /snapshot/

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

# TODO: emit Go SBOM here for the inline-compliance pipeline. When wired:
#   RUN go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@v1.7.0
#   RUN cyclonedx-gomod app -licenses -json -main ./cmd/main.go \
#       -output /sbom-go.cdx.json .
# The SBOM is then COPY'd into the licenses stage (added below) which runs
# python3 -m compliance.generators --ecosystem go to render NOTICES-Go.txt.
#
# This requires:
#   1. Adding --build-context compliance=../../container/compliance to the
#      buildx invocation in .github/actions/build-deploy-component/action.yml
#      so the licenses stage can COPY in the compliance code.
#   2. Adding a `licenses` stage FROM python:3.12-slim that runs the
#      generator + policy validator. (operator's distroless final has no
#      shell or Python; the licenses generation happens in a separate
#      Python-bearing stage and only the resulting /legal tree gets COPY'd
#      into the final.)
#   3. Adding `sboms` and `legal` scratch stages following the same shape
#      as container/templates/dynamo_runtime.Dockerfile.
#   4. Modifying the runtime stage below to COPY --from=licenses /legal /legal
#      so the operator image ships its NOTICES.

# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v4.0.3
WORKDIR /
COPY --from=builder /workspace/manager .
COPY --from=builder /workspace/crd-apply .
COPY --from=builder /workspace/config/crd/bases/ /opt/dynamo-operator/crds/
USER 65532:65532

ENTRYPOINT ["./manager"]
