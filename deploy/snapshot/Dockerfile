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
# Default to upstream CRIU development branch. Custom forks can override both
# args at build time, for example:
#   --build-arg CRIU_REPO=<git-remote-url>
#   --build-arg CRIU_REF=<fork-branch-or-sha>
ARG CRIU_REPO=https://github.com/checkpoint-restore/criu.git
ARG CRIU_REF=criu-dev
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

# Emit a CycloneDX SBOM for snapshot-agent's Go module set. nsrestore's
# deps are a subset of agent's, so a single SBOM keyed on ./cmd/agent is
# sufficient. The git-init dance satisfies cyclonedx-gomod's main-module
# version probe (Docker build context doesn't carry .git). Pinned to
# keep the SBOM tool itself reproducible.
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    --mount=type=cache,target=/root/.cache/go-build,sharing=shared \
    GOBIN=/usr/local/bin go install github.com/CycloneDX/cyclonedx-gomod/cmd/cyclonedx-gomod@v1.7.0 && \
    git init -q && \
    git -c user.email=build@dynamo -c user.name=build add -A && \
    git -c user.email=build@dynamo -c user.name=build commit -q -m sbom && \
    cyclonedx-gomod app -licenses -json -main ./cmd/agent -output /sbom-go.cdx.json .

# Compliance: harvest each Go module's real LICENSE files from the module cache
# so the go NOTICES generator inlines upstream license text instead of canonical
# SPDX fallback. Preserve the cache's escaped "<module-path>@<version>/" layout —
# generators/go.py escapes module names the same way to look up. Non-fatal.
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    mkdir -p /sbom-go-licenses && cd /go/pkg/mod && \
    find . -path ./cache -prune -o -type f \
        \( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'COPYING*' \
           -o -iname 'NOTICE*' -o -iname 'UNLICENSE*' \) -print | \
    while IFS= read -r lf; do \
        rel="${lf#./}"; dest="/sbom-go-licenses/$(dirname "$rel")"; \
        mkdir -p "$dest" && cp "$lf" "$dest/" 2>/dev/null || true; \
    done; \
    echo "go license harvest: $(find /sbom-go-licenses -type f 2>/dev/null | wc -l) files"; true

# Compliance source archival: vendor third-party Go modules. Same pattern
# as operator's builder. First-party (github.com/ai-dynamo/) pruned later
# in collect_sources.py.
ARG ENABLE_SOURCE_ARCHIVAL=false
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        go mod vendor -o /tmp/go-vendor ; \
    else \
        mkdir -p /tmp/go-vendor ; \
    fi

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
# Stage: Agent (pre-compliance) - Everything that ships, minus /legal
# =============================================================================
# Split out so the licenses stage can run dpkg-query against the actual
# installed package state (CRIU runtime libs included), then the final
# `agent` stage is a thin downstream that COPYs /legal in.
FROM ${AGENT_BASE_IMAGE} AS agent_pre

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

# =============================================================================
# Stage: Compliance: licenses
# =============================================================================
# Runs in agent_pre's actual filesystem so dpkg-query enumerates the real
# runtime package set (cuda-dl-base + everything we apt-get installed in
# agent_pre). cyclonedx-gomod produced /sbom-go.cdx.json upstream in the
# Go builder; we COPY it in here for the go generator.
#
# `compliance` build context is plumbed by .github/actions/build-deploy-component/action.yml
# as `--build-context compliance=../../container/compliance`.
FROM agent_pre AS licenses

USER root
RUN mkdir -p /legal /sboms
# Install python3 + python3-yaml. PyYAML is required for the override
# lookup — without it, license_overrides.yaml silently no-ops and
# NVIDIA-proprietary dpkgs (libnccl2 etc.) come back UNKNOWN despite
# having hand-curated overrides.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-yaml \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=compliance . /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=builder /sbom-go.cdx.json /tmp/sbom-go.cdx.json
COPY --from=builder /sbom-go-licenses /tmp/go-licenses

# go + dpkg cover the runtime image's apt + Go module set. native
# enumerates from-source components (CRIU, cuda-checkpoint,
# cuda-checkpoint-helper) — these are attributed by the manual
# `COPY /tmp/criu/COPYING /legal/CRIU/COPYING` lines in agent_pre, which
# the native generator reads via license_text_path to inline in NOTICES.
# BASELINE_SBOM_FILE (optional, empty default): the per-arch baseline SBOM
# *stem* under /opt/compliance/base_sboms/. We append -${TARGETARCH}.cdx.json
# so the matching-arch floor is subtracted (snapshot ships amd64-only today).
ARG BASELINE_SBOM_FILE=""
ARG TARGETARCH
RUN python3 -m compliance.generators \
    --ecosystem go,dpkg,native \
    --go-sbom /tmp/sbom-go.cdx.json \
    --go-licenses-dir /tmp/go-licenses \
    --native-yaml /opt/compliance/native_packages.yaml \
    --native-image snapshot-agent \
    --output-dir /legal \
    --policy /opt/compliance/policy/licenses.toml \
    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
    -v
# Policy gate on the single unified CSV (its `ecosystem` column scopes each row).
RUN python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input /legal/osrb-deps.csv


# =============================================================================
# Stage: Compliance: artifact (single unified -compliance tree for CI extraction)
# =============================================================================
FROM scratch AS compliance_artifact
COPY --from=licenses /legal/ /


# =============================================================================
# Stage: Compliance: sources
# =============================================================================
# Collects third-party Go module sources + preserved native source tarballs
# for OSRB submission. Output is a separate sources zip cross-referenced in
# the OSRB bundle's build-provenance.json; never COPY'd into the runtime.
FROM python:3.12-slim AS sources_collect
WORKDIR /
RUN mkdir -p /sources /opt/compliance /opt/go-vendor /opt/native-sources
COPY --from=compliance . /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=builder /tmp/go-vendor/ /opt/go-vendor/
# Native source trees (CRIU sources + cuda-checkpoint clone) from criu-builder.
#
# These COPYs are unconditional (not gated on ENABLE_SOURCE_ARCHIVAL) by
# design: sources_collect is its own stage that is NEVER COPY'd into the
# runtime agent image — the runtime only consumes /sources.zip via the
# `sources_archive` stage. So copying the source trees in here costs nothing
# in the shipped image. Gating them would risk archive completeness (an ARG
# flip would silently drop CRIU+cuda-checkpoint from the archive).
COPY --from=criu-builder /tmp/criu/ /opt/native-sources/criu/
COPY --from=criu-builder /tmp/cuda-checkpoint/ /opt/native-sources/cuda-checkpoint/

ARG ENABLE_SOURCE_ARCHIVAL=false
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        python3 -m compliance.collect_sources \
            --ecosystem go --ecosystem native \
            --output-zip /sources.zip \
            --sources-root /sources \
            --go-vendor-dir /opt/go-vendor \
            --native-source-dir /opt/native-sources \
            -v ; \
    else \
        : > /sources.zip ; \
    fi


FROM scratch AS sources_archive
COPY --from=sources_collect /sources.zip /sources.zip


# =============================================================================
# Stage: Agent - Final snapshot-agent image
# =============================================================================
FROM agent_pre AS agent
COPY --from=licenses /legal /legal

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
RUN chmod +x /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint-helper

# Copy nsrestore binary (invoked by DaemonSet via nsenter)
COPY --from=builder /nsrestore /usr/local/bin/nsrestore
RUN chmod +x /usr/local/bin/nsrestore

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /var/criu-work
