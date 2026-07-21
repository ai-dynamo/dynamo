# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Base stage - Common setup for all stages
ARG DOCKER_PROXY
ARG BASE_IMAGE="library/golang"
ARG BASE_IMAGE_TAG="1.26.3"
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
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o crd-apply ./cmd/crd-apply/main.go && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath -ldflags="-s -w" -o dgd-apply-overrides ./cmd/dgd-apply-overrides

# Emit a CycloneDX SBOM for the operator's Go module set so the licenses
# stage below can render NOTICES-Go.txt. cyclonedx-gomod is pinned to
# v1.7.0 to keep the SBOM tool itself reproducible.
#
# `app -main ./cmd` covers the manager binary (./cmd/main.go = package
# main in ./cmd). crd-apply and dgd-apply-overrides are separate mains in
# the same module, so their deps are a subset of manager's — a single SBOM
# is sufficient. Tried `mod` first but its license detection is weaker than
# `app`'s — everything came back UNKNOWN.
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

# Compliance source archival: vendor third-party Go modules into /tmp/go-vendor
# for the sources_collect stage. Gated on ENABLE_SOURCE_ARCHIVAL so PR builds
# skip the ~50-150 MB vendor pull. First-party modules (github.com/ai-dynamo/)
# are pruned later in collect_sources.py.
ARG ENABLE_SOURCE_ARCHIVAL=false
RUN --mount=type=cache,target=/go/pkg/mod,sharing=shared \
    if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        go mod vendor -o /tmp/go-vendor ; \
    else \
        mkdir -p /tmp/go-vendor ; \
    fi

#######################################
########## Compliance: licenses #######
#######################################
#
# The operator's runtime image is distroless — no shell, no Python — so the
# generator can't run inside the runtime layer. This stage uses python:3.12-slim
# as a Python-bearing helper, COPYs the SBOM out of `builder` and the compliance
# code in via a named build context, renders NOTICES-Go.txt + go-deps.csv into
# /legal/, validates against licenses.toml, and is then drained into a scratch
# `compliance_artifact` stage that CI extracts and the runtime COPYs from.
#
# `compliance` build context is plumbed by .github/actions/build-deploy-component/action.yml
# as `--build-context compliance=../../container/compliance`.
FROM ${DOCKER_PROXY}library/python:3.12-slim AS licenses

WORKDIR /
RUN mkdir -p /legal /sboms
# Copy the contents of container/compliance/ into /opt/compliance/. With
# PYTHONPATH=/opt, python3 -m compliance.<sub> resolves to /opt/compliance/<sub>/.
COPY --from=compliance . /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=builder /sbom-go.cdx.json /tmp/sbom-go.cdx.json
COPY --from=builder /sbom-go-licenses /tmp/go-licenses

# Only `go` is invoked — the operator runtime is distroless (no dpkg metadata
# survives), ships no Python or Rust at runtime, and has no native sources of
# its own. Base-image components (ca-certificates etc.) are NGC's responsibility
# and tracked separately via the base-SBOM corpus.
#
# BASELINE_SBOM_FILE (optional, empty default): the per-arch baseline SBOM
# *stem* under /opt/compliance/base_sboms/. We append -${TARGETARCH}.cdx.json so
# each platform of this multi-arch image subtracts its own-arch floor. Passed by
# build-deploy-component/action.yml (go@<digest8> for the distroless-go base).
ARG BASELINE_SBOM_FILE=""
ARG TARGETARCH
RUN python3 -m compliance.generators \
    --ecosystem go \
    --go-sbom /tmp/sbom-go.cdx.json \
    --go-licenses-dir /tmp/go-licenses \
    --output-dir /legal \
    --policy /opt/compliance/policy/licenses.toml \
    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
    -v
# Policy gate on the single unified CSV (its `ecosystem` column scopes each row).
RUN python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input /legal/osrb-deps.csv


#######################################
####### Compliance: artifact ##########
#######################################
#
# Single FROM-scratch stage exposing the unified compliance tree for CI
# extraction (one `-compliance` artifact): flat NOTICES-<Eco>.txt + osrb-deps.csv
# + osrb.cdx.json. The runtime stage below COPYs from `licenses` directly; this
# scratch stage exists purely for CI extraction.

FROM scratch AS compliance_artifact
COPY --from=licenses /legal/ /


#######################################
########## Compliance: sources ########
#######################################
#
# Collects third-party Go module sources for OSRB submission. Mirrors the
# `sources_collect` / `sources_archive` pattern used by the runtime
# templates (container/templates/compliance.Dockerfile). Output is a separate
# sources-<image>-<arch>.zip companion archive referenced by the OSRB bundle
# (not bundled in the runtime image).
#
# Uses python:3.12-slim as the helper (operator's distroless final has
# no shell/python). CI extracts via `--target sources_archive` and never
# COPYs into the runtime image.
FROM ${DOCKER_PROXY}library/python:3.12-slim AS sources_collect
WORKDIR /
RUN mkdir -p /sources /opt/compliance /opt/go-vendor
COPY --from=compliance . /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=builder /tmp/go-vendor/ /opt/go-vendor/

ARG ENABLE_SOURCE_ARCHIVAL=false
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        python3 -m compliance.collect_sources \
            --ecosystem go \
            --output-zip /sources.zip \
            --sources-root /sources \
            --go-vendor-dir /opt/go-vendor \
            -v ; \
    else \
        python3 -c "import zipfile; zipfile.ZipFile('/sources.zip','w').close()" ; \
    fi


FROM scratch AS sources_archive
COPY --from=sources_collect /sources.zip /sources.zip


# Runtime stage
FROM nvcr.io/nvidia/distroless/go:v4.0.3
WORKDIR /
COPY --from=builder /workspace/manager .
COPY --from=builder /workspace/crd-apply .
COPY --from=builder /workspace/dgd-apply-overrides .
COPY --from=builder /workspace/config/crd/bases/ /opt/dynamo-operator/crds/
COPY --from=licenses /legal /legal
USER 65532:65532

ENTRYPOINT ["./manager"]
