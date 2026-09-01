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
# the well-known protos the engine .proto files import (vllm_grpc.proto pulls
# in google/protobuf/struct.proto); utoipa-swagger-ui downloads its pinned web
# asset through the system curl binary.
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

# ---- vLLM builder -----------------------------------------------------------
FROM builder-base AS vllm-builder

ARG TARGETARCH

# The three builders share one target cache per architecture, so each backend
# reuses the dependency artifacts the previous one compiled. sharing=locked
# serializes them, which is required for correctness — cargo's .package-cache
# lock lives in $CARGO_HOME, outside these mounts, so it cannot coordinate
# separate build containers — and is also what bounds peak memory, since
# codegen-units=1 plus LTO makes each release link expensive on its own.
# Every id is per-architecture so the amd64 and arm64 halves of a multi-platform
# build do not serialize against each other.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-vllm-sidecar \
    && strip target/release/dynamo-vllm-sidecar \
    && install -D target/release/dynamo-vllm-sidecar /out/dynamo-vllm-sidecar

# ---- SGLang builder ---------------------------------------------------------
FROM builder-base AS sglang-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-sglang-sidecar \
    && strip target/release/dynamo-sglang-sidecar \
    && install -D target/release/dynamo-sglang-sidecar /out/dynamo-sglang-sidecar

# ---- TensorRT-LLM builder ---------------------------------------------------
FROM builder-base AS trtllm-builder

ARG TARGETARCH

RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-target-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p dynamo-trtllm-sidecar \
    && strip target/release/dynamo-trtllm-sidecar \
    && install -D target/release/dynamo-trtllm-sidecar /out/dynamo-trtllm-sidecar

# ---- compliance inventory ----------------------------------------------------
FROM builder-base AS compliance-inventory

ARG TARGETARCH
ARG CARGO_CYCLONEDX_VERSION=0.5.9

# Runtime images get cargo-cyclonedx output through maturin's wheel SBOMs. The
# sidecars are standalone binaries, so run the same upstream generator directly
# and stage its output where the common Rust compliance generator can read it.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-tools-${TARGETARCH},target=/tmp/cargo-tools,sharing=locked \
    CARGO_TARGET_DIR=/tmp/cargo-tools \
    cargo install cargo-cyclonedx --locked --version "${CARGO_CYCLONEDX_VERSION}"

# Generate one target-specific SBOM per shipped executable. Development and
# build-only dependencies cannot be linked into these binaries and are omitted.
RUN --mount=type=cache,id=dynamo-sidecar-cargo-registry-${TARGETARCH},target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=dynamo-sidecar-cargo-git-${TARGETARCH},target=/usr/local/cargo/git,sharing=locked \
    set -e; \
    RUSTARCH=$(case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) echo "x86_64-unknown-linux-gnu" ;; \
        arm64) echo "aarch64-unknown-linux-gnu" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH:-unset}" >&2; exit 1 ;; \
    esac) \
    && mkdir -p /out/sidecar-sboms /out/rust-licenses \
    && for spec in \
        "lib/sidecar/vllm/Cargo.toml:dynamo-vllm-sidecar" \
        "lib/sidecar/sglang/Cargo.toml:dynamo-sglang-sidecar" \
        "lib/sidecar/trtllm/Cargo.toml:dynamo-trtllm-sidecar"; do \
        manifest="${spec%%:*}"; \
        binary="${spec##*:}"; \
        cargo cyclonedx \
            --manifest-path "${manifest}" \
            --format json \
            --spec-version 1.5 \
            --describe binaries \
            --target "${RUSTARCH}" \
            --no-build-deps; \
        install -m 0644 \
            "$(dirname "${manifest}")/${binary}_bin.cdx.json" \
            "/out/sidecar-sboms/${binary}.cyclonedx.json"; \
    done \
    && test -s /out/sidecar-sboms/dynamo-vllm-sidecar.cyclonedx.json \
    && test -s /out/sidecar-sboms/dynamo-sglang-sidecar.cyclonedx.json \
    && test -s /out/sidecar-sboms/dynamo-trtllm-sidecar.cyclonedx.json \
    && for src in /usr/local/cargo/registry/src/*; do \
        [ -d "${src}" ] || continue; \
        find "${src}" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r crate; do \
            dest="/out/rust-licenses/$(basename "${crate}")"; \
            for license in "${crate}"/LICENSE* "${crate}"/LICENCE* "${crate}"/COPYING* "${crate}"/NOTICE* "${crate}"/UNLICENSE*; do \
                [ -f "${license}" ] || continue; \
                mkdir -p "${dest}" && cp "${license}" "${dest}/"; \
            done; \
        done; \
    done

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

COPY --from=vllm-builder /out/dynamo-vllm-sidecar /usr/local/bin/
COPY --from=sglang-builder /out/dynamo-sglang-sidecar /usr/local/bin/
COPY --from=trtllm-builder /out/dynamo-trtllm-sidecar /usr/local/bin/

COPY --chmod=755 lib/sidecar/dynamo-sidecar /usr/local/bin/dynamo-sidecar

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
RUN mkdir -p /legal /tmp/runtime-root \
    /tmp/sidecar-site/dynamo_sidecar.dist-info/sboms

# Scan the filesystem that actually ships, rather than the build-tool packages
# present in this helper stage. The Rust SBOM contains only dependencies linked
# into the three sidecar executables.
COPY --from=runtime-base / /tmp/runtime-root/
COPY --from=compliance-inventory /out/sidecar-sboms/ \
    /tmp/sidecar-site/dynamo_sidecar.dist-info/sboms/
COPY --from=compliance-inventory /out/rust-licenses /tmp/rust-licenses

ENV PYTHONPATH=/src/container
RUN python3 -m compliance.generators \
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

# ---- shipped runtime ---------------------------------------------------------
FROM runtime-base AS runtime

# The same policy-gated NOTICES and CycloneDX/OSRB inventory uploaded by CI are
# included in the image for downstream consumers.
COPY --from=licenses /legal /legal
