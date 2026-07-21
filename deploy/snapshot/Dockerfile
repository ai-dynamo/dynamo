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
ARG CRIU_REPO=https://github.com/checkpoint-restore/criu.git
ARG CRIU_REF=b47c692bb3a73ae45ee1c4ab215a8f6869d40efd
ARG CRIU_SHA=
ARG CUDA_CHECKPOINT_REPO=https://github.com/NVIDIA/cuda-checkpoint.git
ARG CUDA_CHECKPOINT_REF=
ARG CUDA_CHECKPOINT_SHA=
ARG AGENT_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
ARG DYNAMO_COMMIT_SHA

# For placeholder target only - this default allows agent builds to succeed,
# but placeholder builds MUST override it with --build-arg BASE_IMAGE=<image>
ARG BASE_IMAGE=placeholder-requires-base-image-arg
ARG BASE_IMAGE_USER=root

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
ARG CRIU_SHA
ARG CUDA_CHECKPOINT_REPO
ARG CUDA_CHECKPOINT_REF
ARG CUDA_CHECKPOINT_SHA
ARG DYNAMO_COMMIT_SHA

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
    && git remote add origin "${CRIU_REPO}" \
    && if [ -n "${CRIU_SHA}" ] && [ "${CRIU_REF}" != "${CRIU_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CRIU_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CRIU_SHA}"; \
       fi \
    && git fetch --depth 1 origin "${CRIU_SHA:-${CRIU_REF}}" \
    && git checkout --detach FETCH_HEAD \
    && { [ -z "${CRIU_SHA}" ] || test "$(git rev-parse HEAD)" = "${CRIU_SHA}"; } \
    && make -j$(nproc) \
    && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin

COPY plugins/inet-remap/snapshot_inet_remap.c /tmp/snapshot_inet_remap.c
RUN mkdir -p /criu-install/usr/local/lib/snapshot/criu-plugins \
    && gcc -O2 -Wall -Wextra -fPIC -shared \
      -idirafter /tmp/criu/criu/include \
      -o /criu-install/usr/local/lib/snapshot/criu-plugins/snapshot_inet_remap.so \
      /tmp/snapshot_inet_remap.c

RUN git init /tmp/cuda-checkpoint \
    && cd /tmp/cuda-checkpoint \
    && git remote add origin "${CUDA_CHECKPOINT_REPO}" \
    && if [ -n "${CUDA_CHECKPOINT_SHA}" ] && [ "${CUDA_CHECKPOINT_REF}" != "${CUDA_CHECKPOINT_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CUDA_CHECKPOINT_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CUDA_CHECKPOINT_SHA}"; \
       fi \
    && if [ -n "${CUDA_CHECKPOINT_SHA}" ]; then \
         git fetch --depth 1 origin "${CUDA_CHECKPOINT_SHA}" \
         && git checkout --detach FETCH_HEAD \
         && test "$(git rev-parse HEAD)" = "${CUDA_CHECKPOINT_SHA}"; \
       else \
         git fetch --depth 1 origin "${CUDA_CHECKPOINT_REF:-HEAD}" \
         && git checkout --detach FETCH_HEAD; \
       fi

RUN { [ -z "${DYNAMO_COMMIT_SHA}" ] || printf '%s\n' "${DYNAMO_COMMIT_SHA}" | grep -Eq '^[0-9a-f]{40}$'; } \
    && printf '%s\n' \
      "dynamo_source_sha=${DYNAMO_COMMIT_SHA}" \
      "criu_source_url=${CRIU_REPO}" \
      "criu_source_ref=${CRIU_REF}" \
      "criu_source_sha=${CRIU_SHA}" \
      "cuda_checkpoint_source_url=${CUDA_CHECKPOINT_REPO}" \
      "cuda_checkpoint_source_ref=${CUDA_CHECKPOINT_REF}" \
      "cuda_checkpoint_source_sha=${CUDA_CHECKPOINT_SHA}" \
      > /restore-tool-source-provenance.txt

# =============================================================================
# Stage: Agent (pre-compliance) - Everything that ships, minus /legal
# =============================================================================
# Split out so the licenses stage can run dpkg-query against the actual
# installed package state (CRIU runtime libs included), then the final
# `agent` stage is a thin downstream that COPYs /legal in.
FROM ${AGENT_BASE_IMAGE} AS agent_pre

ARG TARGETARCH=amd64
ARG DYNAMO_COMMIT_SHA
ARG CRIU_SHA
ARG CUDA_CHECKPOINT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}
LABEL org.opencontainers.image.revision=${DYNAMO_COMMIT_SHA} \
    ai.dynamo.criu.source-sha=${CRIU_SHA} \
    ai.dynamo.cuda-checkpoint.source-sha=${CUDA_CHECKPOINT_SHA}

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
COPY --from=criu-builder /restore-tool-source-provenance.txt /opt/dynamo/restore-tool-source-provenance.txt
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
# Freeze the shipped dpkg database and copyright files before installing
# compliance-only helpers. The final image is FROM agent_pre, so packages
# installed below must not appear in dpkg NOTICES/OSRB output.
RUN mkdir -p /runtime-root/var/lib /runtime-root/usr/share \
    && cp -a /var/lib/dpkg /runtime-root/var/lib/ \
    && cp -a /usr/share/doc /runtime-root/usr/share/
# Install python3 + python3-yaml for the compliance tooling only. PyYAML is
# required for license_overrides.yaml; dpkg scanning reads /runtime-root so
# these helper packages are not attributed as shipped content.
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
    --dpkg-root /runtime-root \
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
# Collects source archives for dpkg deltas, third-party Go modules, and
# preserved native source trees for OSRB submission. Output is a separate
# sources zip cross-referenced in the OSRB bundle's build-provenance.json;
# never COPY'd into the runtime.
FROM agent_pre AS sources_collect
WORKDIR /
RUN mkdir -p /sources /opt/compliance /opt/go-vendor /opt/native-sources /runtime-root/var/lib \
    && cp -a /var/lib/dpkg /runtime-root/var/lib/
# Python is needed only to run compliance.collect_sources. The dpkg package
# set comes from /runtime-root, captured before this install.
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
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
ARG BASELINE_SBOM_FILE=""
ARG TARGETARCH
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        python3 -m compliance.collect_sources \
            --ecosystem dpkg --ecosystem go --ecosystem native \
            --output-zip /sources.zip \
            --sources-root /sources \
            --dpkg-root /runtime-root \
            --go-vendor-dir /opt/go-vendor \
            --native-source-dir /opt/native-sources \
            ${BASELINE_SBOM_FILE:+--baseline-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
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

# Restore tools for the pinned vLLM nightly's Jammy ABI. These stages are used
# only by the glm52-placeholder target below; the canonical placeholder target
# retains its current Noble composition.
FROM ${BASE_IMAGE} AS cuda-helper-builder-placeholder

USER root

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

FROM ubuntu:22.04 AS criu-builder-placeholder

ARG CRIU_REPO
ARG CRIU_REF
ARG CRIU_SHA
ARG CUDA_CHECKPOINT_REPO
ARG CUDA_CHECKPOINT_REF
ARG CUDA_CHECKPOINT_SHA
ARG DYNAMO_COMMIT_SHA

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
    python3-pip \
    python3-protobuf \
    libgnutls28-dev \
    libnftables-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init /tmp/criu \
    && cd /tmp/criu \
    && git remote add origin "${CRIU_REPO}" \
    && if [ -n "${CRIU_SHA}" ] && [ "${CRIU_REF}" != "${CRIU_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CRIU_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CRIU_SHA}"; \
       fi \
    && git fetch --depth 1 origin "${CRIU_SHA:-${CRIU_REF}}" \
    && git checkout --detach FETCH_HEAD \
    && { [ -z "${CRIU_SHA}" ] || test "$(git rev-parse HEAD)" = "${CRIU_SHA}"; } \
    && make -j$(nproc) \
    && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin

COPY plugins/inet-remap/snapshot_inet_remap.c /tmp/snapshot_inet_remap.c
RUN mkdir -p /criu-install/usr/local/lib/snapshot/criu-plugins \
    && gcc -O2 -Wall -Wextra -fPIC -shared \
      -idirafter /tmp/criu/criu/include \
      -o /criu-install/usr/local/lib/snapshot/criu-plugins/snapshot_inet_remap.so \
      /tmp/snapshot_inet_remap.c

RUN git init /tmp/cuda-checkpoint \
    && cd /tmp/cuda-checkpoint \
    && git remote add origin "${CUDA_CHECKPOINT_REPO}" \
    && if [ -n "${CUDA_CHECKPOINT_SHA}" ] && [ "${CUDA_CHECKPOINT_REF}" != "${CUDA_CHECKPOINT_SHA}" ]; then \
         test "$(git ls-remote --exit-code origin "${CUDA_CHECKPOINT_REF}" | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }')" = "${CUDA_CHECKPOINT_SHA}"; \
       fi \
    && if [ -n "${CUDA_CHECKPOINT_SHA}" ]; then \
         git fetch --depth 1 origin "${CUDA_CHECKPOINT_SHA}" \
         && git checkout --detach FETCH_HEAD \
         && test "$(git rev-parse HEAD)" = "${CUDA_CHECKPOINT_SHA}"; \
       else \
         git fetch --depth 1 origin "${CUDA_CHECKPOINT_REF:-HEAD}" \
         && git checkout --detach FETCH_HEAD; \
       fi

RUN { [ -z "${DYNAMO_COMMIT_SHA}" ] || printf '%s\n' "${DYNAMO_COMMIT_SHA}" | grep -Eq '^[0-9a-f]{40}$'; } \
    && printf '%s\n' \
      "dynamo_source_sha=${DYNAMO_COMMIT_SHA}" \
      "criu_source_url=${CRIU_REPO}" \
      "criu_source_ref=${CRIU_REF}" \
      "criu_source_sha=${CRIU_SHA}" \
      "cuda_checkpoint_source_url=${CUDA_CHECKPOINT_REPO}" \
      "cuda_checkpoint_source_ref=${CUDA_CHECKPOINT_REF}" \
      "cuda_checkpoint_source_sha=${CUDA_CHECKPOINT_SHA}" \
      > /restore-tool-source-provenance.txt

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

# =============================================================================
# GLM 5.2 pinned-nightly placeholder
# =============================================================================
FROM ${BASE_IMAGE} AS glm52-placeholder

ARG BASE_IMAGE
ARG BASE_IMAGE_USER
ARG DYNAMO_COMMIT_SHA
ARG CRIU_SHA
ARG CUDA_CHECKPOINT_SHA
ARG TARGETARCH=amd64
ENV ORIGINAL_BASE_IMAGE=${BASE_IMAGE}
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}
LABEL org.opencontainers.image.revision=${DYNAMO_COMMIT_SHA} \
    ai.dynamo.criu.source-sha=${CRIU_SHA} \
    ai.dynamo.cuda-checkpoint.source-sha=${CUDA_CHECKPOINT_SHA}

USER root

RUN if [ "${TARGETARCH}" != "amd64" ]; then \
      echo "ERROR: Dynamo Snapshot requires x86_64 (cuda-checkpoint has no ${TARGETARCH} binary)" >&2; exit 1; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    libbsd0 \
    libcap2 \
    libnet1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libprotobuf-c1 \
    libgnutls30 \
    libnftables1 \
    iproute2 \
    iptables \
    procps \
    uuid-runtime \
    tar \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=criu-builder-placeholder /criu-install/usr/local /usr/local
COPY --from=criu-builder-placeholder /tmp/criu/COPYING /legal/CRIU/COPYING
COPY --from=criu-builder-placeholder /restore-tool-source-provenance.txt /opt/dynamo/restore-tool-source-provenance.txt

COPY --from=criu-builder-placeholder /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
COPY --from=criu-builder-placeholder /tmp/cuda-checkpoint/LICENSE /legal/cuda-checkpoint/LICENSE
COPY --from=cuda-helper-builder-placeholder /cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
RUN chmod +x /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint-helper

COPY --from=builder /nsrestore /usr/local/bin/nsrestore
RUN chmod +x /usr/local/bin/nsrestore

# Fail the image build if any restore-tool binary has the wrong runtime ABI.
# cuda-checkpoint reaches main with the build-only CUDA stub, then exits 255
# because the real NVIDIA driver is intentionally not present during builds.
RUN criu --version \
    && test -f /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so \
    && cuda_stub_dir="$(mktemp -d)" \
    && ln -s /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so "${cuda_stub_dir}/libcuda.so.1" \
    && if LD_LIBRARY_PATH="${cuda_stub_dir}" cuda-checkpoint --help >/dev/null 2>"${cuda_stub_dir}/cuda-checkpoint.stderr"; then \
         :; \
       else \
         test "$?" -eq 255 \
         && grep -Fxq "Insufficient driver" "${cuda_stub_dir}/cuda-checkpoint.stderr"; \
       fi \
    && LD_LIBRARY_PATH="${cuda_stub_dir}" cuda-checkpoint-helper --help >/dev/null \
    && rm -rf "${cuda_stub_dir}" \
    && nsrestore -h >/dev/null 2>&1 \
    && echo "Snapshot restore tools installed successfully"

# Create directories
RUN mkdir -p /checkpoints /var/run/criu /var/criu-work

USER ${BASE_IMAGE_USER}
