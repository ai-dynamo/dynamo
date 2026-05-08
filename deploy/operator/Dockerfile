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

# Emit a CycloneDX SBOM for the operator's Go module set so the licenses
# stage below can render NOTICES-Go.txt. cyclonedx-gomod is pinned to
# v1.7.0 to keep the SBOM tool itself reproducible.
#
# `app -main ./cmd` covers the manager binary (./cmd/main.go = package
# main in ./cmd). crd-apply's deps are a subset of manager's, so a
# single SBOM is sufficient. Tried `mod` first but its license detection
# is weaker than `app`'s — everything came back UNKNOWN.
# cyclonedx-gomod resolves the main module's own version via `go list -m`,
# which falls back to git. The build context doesn't carry .git (Docker
# build context is the per-component dir), so a minimal in-stage git init
# satisfies it. The recorded version is "(devel)" — fine, since the SBOM's
# purpose here is the dependency components, not the operator's own entry.
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    --mount=type=cache,target=/root/.cache/go-build,sharing=shared \
    GOBIN=/usr/local/bin go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@v1.7.0 && \
    git init -q && \
    git -c user.email=build@dynamo -c user.name=build add -A && \
    git -c user.email=build@dynamo -c user.name=build commit -q -m sbom && \
    cyclonedx-gomod app -licenses -json -main ./cmd -output /sbom-go.cdx.json .

#######################################
########## Compliance: licenses #######
#######################################
#
# The operator's runtime image is distroless — no shell, no Python — so the
# generator can't run inside the runtime layer. This stage uses python:3.12-slim
# as a Python-bearing helper, COPYs the SBOM out of `builder` and the compliance
# code in via a named build context, renders NOTICES-Go.txt + go-deps.csv into
# /legal/, validates against licenses.toml, and is then drained into a scratch
# `legal` stage that the runtime COPYs from.
#
# `compliance` build context is plumbed by .github/actions/build-deploy-component/action.yml
# as `--build-context compliance=../../container/compliance`.
FROM python:3.12-slim AS licenses

WORKDIR /
RUN mkdir -p /legal /sboms
# Copy the contents of container/compliance/ into /opt/compliance/. With
# PYTHONPATH=/opt, python3 -m compliance.<sub> resolves to /opt/compliance/<sub>/.
COPY --from=compliance . /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=builder /sbom-go.cdx.json /tmp/sbom-go.cdx.json

# Only `go` is invoked — the operator runtime is distroless (no dpkg metadata
# survives), ships no Python or Rust at runtime, and has no native sources of
# its own. Base-image components (ca-certificates etc.) are NGC's responsibility
# and tracked separately via the base-SBOM corpus.
RUN python3 -m compliance.generators \
    --ecosystem go \
    --go-sbom /tmp/sbom-go.cdx.json \
    --output-dir /legal \
    -v
RUN find /legal -name '*-deps.csv' -print0 | \
    xargs -0 -n1 -I {} python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input {}
# Move *-deps.csv files OUT of /legal/ so only NOTICES + license text ship in
# the runtime image. CSVs go to /sboms/ for CI extraction (--target sboms).
RUN find /legal -name '*-deps.csv' -print -exec sh -c \
    'mkdir -p "/sboms/$(basename $(dirname "$1"))" && mv "$1" "/sboms/$(basename $(dirname "$1"))/$(basename "$1")"' _ {} \;


#######################################
########## Compliance: sboms ##########
#######################################
#
# CI extracts via `docker buildx build --target sboms --output type=local,dest=...`
# to archive per-ecosystem deps CSVs. NEVER COPY'd into the runtime image.

FROM scratch AS sboms
COPY --from=licenses /sboms/ /


#######################################
########## Compliance: legal ##########
#######################################
#
# FROM scratch so CI can extract just the NOTICES tree without exporting the
# multi-GB licenses-stage filesystem. The runtime stage below COPYs from
# `licenses` directly (not from `legal`); the scratch stage exists purely
# for CI extraction shape consistency with the runtime templates.

FROM scratch AS legal
COPY --from=licenses /legal/ /


# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v4.0.3
WORKDIR /
COPY --from=builder /workspace/manager .
COPY --from=builder /workspace/crd-apply .
COPY --from=builder /workspace/config/crd/bases/ /opt/dynamo-operator/crds/
COPY --from=licenses /legal /legal
USER 65532:65532

ENTRYPOINT ["./manager"]
