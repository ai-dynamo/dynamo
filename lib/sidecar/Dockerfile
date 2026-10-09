# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only image containing the vLLM, SGLang, and TensorRT-LLM Rust sidecars.
# The dynamo-sidecar entrypoint selects an engine-specific binary; the GPU
# engine continues to run in a separate container.
#
# Build from the repository root:
#   docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:dev .

# Use NVIDIA's public, signed, multi-architecture Ubuntu base for both build
# and runtime stages. The sidecars are CPU-only and need no framework or CUDA
# runtime image. Keep this tag synchronized with the captured Ubuntu SBOM in
# container/compliance/base_sboms/manifest.json; refresh that baseline before
# advancing the image tag.
ARG BASE_IMAGE=nvcr.io/nvidia/base/ubuntu:noble-20250619

# ---- common builder ---------------------------------------------------------
FROM ${BASE_IMAGE} AS builder-base

ARG TARGETARCH

# Superset of the build dependencies required by all sidecar crates:
# aws-lc-sys/ring need cmake and a C toolchain (plus perl); bindgen needs
# libclang; tonic-build (build.rs) needs protoc, and libprotobuf-dev supplies
# the well-known protos the engine .proto files import. utoipa-swagger-ui
# downloads its pinned web asset through the system curl binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        clang \
        curl \
        git \
        libclang-dev \
        protobuf-compiler \
        libprotobuf-dev \
        perl \
        pkg-config \
        python3 \
        python3-yaml \
        wget \
    && rm -rf /var/lib/apt/lists/*

# NGC does not publish a Rust toolchain image. Keep this setup aligned with
# container/templates/dynamo_base.Dockerfile, which installs Rust for the
# backend containers.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.96.1

RUN ARCH_ALT=$(case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) echo "x86_64" ;; \
        arm64) echo "aarch64" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH:-unset}" >&2; exit 1 ;; \
    esac) && \
    RUSTARCH="${ARCH_ALT}-unknown-linux-gnu" && \
    wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

WORKDIR /src

# Cargo resolves every sidecar through the root workspace and their shared
# local crates. Copy the workspace inputs once instead of maintaining a
# duplicated transitive dependency list for each engine.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .cargo/ .cargo/
COPY lib/ lib/
COPY examples/router/custom-policy-example/ examples/router/custom-policy-example/
COPY deploy/inference-gateway/ext-proc/ deploy/inference-gateway/ext-proc/
COPY deploy/inference-gateway/sidecar/ deploy/inference-gateway/sidecar/

# ---- sidecar builder --------------------------------------------------------
FROM builder-base AS sidecar-builder

ARG TARGETARCH
ARG CARGO_CYCLONEDX_VERSION=0.5.9

# Build all three sidecars and their compliance inventory in one RUN. BuildKit
# takes one of its max-parallelism slots before a RUN waits for a
# sharing=locked cache mount, so every separate RUN on these caches can hold a
# slot idle while another one compiles. One RUN takes the locks once.
#
# sharing=locked is still required: cargo's .package-cache lock lives in
# $CARGO_HOME, outside these mounts, so it cannot coordinate concurrent builds
# of this Dockerfile on the same BuildKit daemon. Running the builds one after
# another also bounds peak memory, since codegen-units=1 plus LTO makes each
# release link expensive. Every id is per-architecture so the amd64 and arm64
# halves of a multi-platform build do not serialize against each other.
#
# Each sidecar keeps its own `cargo build -p` call. One
# `cargo build -p a -p b -p c` would unify features and link the vLLM-only
# features into the SGLang and TensorRT-LLM binaries.
# The target directory exists only during this RUN. The final step removes it
# so the intermediate layer retains only the installed binaries and metadata.
#
# Development and build-only dependencies cannot be linked into these binaries
# and are omitted from the SBOMs.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-tools-${TARGETARCH},target=/tmp/cargo-tools,sharing=locked \
    set -e; \
    RUSTARCH=$(case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) echo "x86_64-unknown-linux-gnu" ;; \
        arm64) echo "aarch64-unknown-linux-gnu" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH:-unset}" >&2; exit 1 ;; \
    esac); \
    CARGO_TARGET_DIR=/tmp/cargo-tools \
        cargo install cargo-cyclonedx --locked --version "${CARGO_CYCLONEDX_VERSION}"; \
    mkdir -p /out/bin /out/sidecar-sboms /out/rust-licenses; \
    for pkg in dynamo-vllm-sidecar dynamo-sglang-sidecar dynamo-trtllm-sidecar; do \
        cargo build --release --locked -p "${pkg}"; \
        strip "target/release/${pkg}"; \
        install -D "target/release/${pkg}" "/out/bin/${pkg}"; \
    done; \
    for pkg in dynamo-vllm-sidecar dynamo-sglang-sidecar dynamo-trtllm-sidecar; do \
        manifest="lib/sidecar/${pkg#dynamo-}"; \
        manifest="${manifest%-sidecar}/Cargo.toml"; \
        cargo cyclonedx \
            --manifest-path "${manifest}" \
            --format json \
            --spec-version 1.5 \
            --describe binaries \
            --target "${RUSTARCH}" \
            --no-build-deps; \
        install -m 0644 \
            "$(dirname "${manifest}")/${pkg}_bin.cdx.json" \
            "/out/sidecar-sboms/${pkg}.cyclonedx.json"; \
        test -s "/out/sidecar-sboms/${pkg}.cyclonedx.json"; \
    done; \
    for src in /usr/local/cargo/registry/src/*; do \
        [ -d "${src}" ] || continue; \
        find "${src}" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r crate; do \
            dest="/out/rust-licenses/$(basename "${crate}")"; \
            for license in "${crate}"/LICENSE* "${crate}"/LICENCE* "${crate}"/COPYING* "${crate}"/NOTICE* "${crate}"/UNLICENSE*; do \
                [ -f "${license}" ] || continue; \
                mkdir -p "${dest}" && cp "${license}" "${dest}/"; \
            done; \
        done; \
    done; \
    rm -rf target

# ---- runtime base ------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime-base

# ca-certificates supports tokenizer downloads and TLS connections to Dynamo
# services. TLS crypto is statically linked, so no engine or GPU libraries are
# needed in this image. The dynamo user uses group 0 for OpenShift
# compatibility and has a home directory for the Hugging Face tokenizer cache.
# The base ships its own uid-1000 `ubuntu` user, so drop it to free the uid.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && { userdel -r ubuntu 2>/dev/null || true; } \
    && useradd --create-home --uid 1000 --gid 0 dynamo

# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so a cache-busted
# rebuild picks up the newest patch level (BuildKit reuses this layer otherwise).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends --only-upgrade \
        dirmngr \
        gnupg \
        gnupg-utils \
        gnupg2 \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        keyboxd \
        libssl3t64 \
        openssl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The entrypoint script comes first so it does not wait behind the compile.
COPY --chmod=755 lib/sidecar/dynamo-sidecar /usr/local/bin/dynamo-sidecar
COPY --from=sidecar-builder /out/bin/ /usr/local/bin/

# Numeric UID so Kubernetes runAsNonRoot can verify it.
USER 1000

# Dynamo system endpoints (/live, /health) — the operator sets DYN_SYSTEM_PORT=9090.
EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/dynamo-sidecar"]
CMD ["--help"]

# ---- compliance --------------------------------------------------------------
FROM builder-base AS licenses

ARG TARGETARCH

USER root
COPY container/compliance/ /src/container/compliance/

# Scan the filesystem that actually ships, rather than the build-tool packages
# present in this helper stage. The Rust SBOM contains only dependencies linked
# into the three sidecar executables. Bind mounts instead of COPY steps keep
# this to one BuildKit op after the runtime image is assembled.
ENV PYTHONPATH=/src/container
RUN --mount=type=bind,from=runtime-base,target=/tmp/runtime-root \
    --mount=type=bind,from=sidecar-builder,source=/out/sidecar-sboms,target=/tmp/sidecar-site/dynamo_sidecar.dist-info/sboms \
    --mount=type=bind,from=sidecar-builder,source=/out/rust-licenses,target=/tmp/rust-licenses \
    mkdir -p /legal \
    && python3 -m compliance.generators \
        --ecosystem rust,dpkg \
        --site-packages /tmp/sidecar-site \
        --dpkg-root /tmp/runtime-root \
        --rust-licenses-dir /tmp/rust-licenses \
        --output-dir /legal \
        --policy /src/container/compliance/policy/licenses.toml \
        --subtract-sbom /src/container/compliance/base_sboms/ubuntu@7291df36-${TARGETARCH}.cdx.json \
        --verbose \
    && python3 -m compliance.policy.validate \
        --policy /src/container/compliance/policy/licenses.toml \
        --input /legal/osrb-deps.csv

FROM scratch AS compliance_artifact
COPY --from=licenses /legal/ /

# ---- CI-only: sidecar binaries as a buildx local-output target --------------
# CI extracts these via `docker buildx build --target sidecar-binaries
# --output=type=local`, not `docker pull` + `docker create` + `docker cp`: the
# builder runners route to a remote BuildKit worker with no local docker
# daemon (see shared-build-sidecar.yml). Not used for any shipped image.
FROM scratch AS sidecar-binaries
COPY --from=sidecar-builder /out/bin/ /

# ---- shipped runtime ---------------------------------------------------------
FROM runtime-base AS runtime

# The same policy-gated NOTICES and CycloneDX/OSRB inventory uploaded by CI are
# included in the image for downstream consumers.
COPY --from=licenses /legal /legal
