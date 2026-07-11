# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12.13-slim-bookworm@sha256:49f3bd3ed3a1c554a43f1e9d03ff303716afed6eda0f3e98c34cd976c8880cd6 AS debian-snapshot

ENV DEBIAN_SNAPSHOT=20260623T000000Z

RUN sed -i \
        -e "s|http://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" \
        -e "s|http://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" \
        /etc/apt/sources.list.d/debian.sources

FROM debian-snapshot AS source-verifier

ARG GIT_REVISION

RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY source.bundle /tmp/source.bundle
RUN GIT_LFS_SKIP_SMUDGE=1 git clone /tmp/source.bundle /source \
    && test "$(git -C /source rev-parse HEAD)" = "${GIT_REVISION}" \
    && test "$(git -C /source rev-parse --is-shallow-repository)" = false \
    && git -C /source fsck --strict \
    && test -z "$(git -C /source status --porcelain)"

FROM nvcr.io/nvidian/dynamo-dev/biswa:prime-p4-83a1c54f-d99e6775-dspark-fi0613-vllm2c17d33f4@sha256:f2630341da92fa7841a86d684b9e58c9b940e3193f9c6062571ea45e37d38b07 AS uv-provider

FROM rust:1.96.1-slim-bookworm@sha256:e18a79fc84dfcfc3ab5ba72290398a644c135c97eaa881447fddc354ee4701a3 AS rust-provider

FROM python:3.12.13-slim-bookworm@sha256:49f3bd3ed3a1c554a43f1e9d03ff303716afed6eda0f3e98c34cd976c8880cd6 AS dynamo-builder

ARG GIT_REVISION
ENV DEBIAN_SNAPSHOT=20260623T000000Z
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}

COPY --from=rust-provider /usr/local/rustup /usr/local/rustup
COPY --from=rust-provider /usr/local/cargo /usr/local/cargo

RUN sed -i \
        -e "s|http://deb.debian.org/debian-security|https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|g" \
        -e "s|http://deb.debian.org/debian|https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|g" \
        /etc/apt/sources.list.d/debian.sources

RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libclang-dev \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /build-venv \
    && cargo install --locked --version 1.9.4 maturin

WORKDIR /build
COPY --from=source-verifier /source/ .
COPY --from=uv-provider /usr/local/bin/uv /usr/local/bin/uv

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/build/lib/bindings/python/target,sharing=locked \
    test "$(git rev-parse HEAD)" = "${GIT_REVISION}" \
    && test -z "$(git status --porcelain)" \
    && cd lib/bindings/python \
    && maturin build --release --interpreter /build-venv/bin/python --out /out \
    && cd /build \
    && uv pip install --python /build-venv/bin/python --no-cache \
        -r benchmarks/router/kubernetes/valkey_sweep/pylock.build.toml \
    && /build-venv/bin/python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /out .

COPY --from=source-verifier /source/benchmarks/router/kubernetes/valkey_sweep/pylock.crick-build.toml /tmp/pylock.crick-build.toml
ADD --checksum=sha256:973b8315fdd72bdeb5fdf4d6b2f444753fc0ebd6380f38f8e1138f8ff8797d99 \
    https://files.pythonhosted.org/packages/a4/d3/8a60c78612fc2d708b94dac7a2dbef4c0de048d64565336a2c0b53c7a872/crick-0.0.8.tar.gz \
    /tmp/crick-0.0.8.tar.gz
RUN uv pip install --python /build-venv/bin/python --no-cache \
        -r /tmp/pylock.crick-build.toml \
    && /build-venv/bin/python -m pip wheel --no-build-isolation --no-deps \
        --wheel-dir /out /tmp/crick-0.0.8.tar.gz

FROM debian-snapshot AS valkey-builder

ENV VALKEY_GIT_REVISION=5b690cefd6cad707a748879c2bab6b72e18efcb7

RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git init /valkey \
    && git -C /valkey remote add origin https://github.com/valkey-io/valkey.git \
    && git -C /valkey fetch --depth=1 origin "${VALKEY_GIT_REVISION}" \
    && git -C /valkey checkout --detach FETCH_HEAD \
    && test "$(git -C /valkey rev-parse HEAD)" = "${VALKEY_GIT_REVISION}" \
    && make -C /valkey -j"$(nproc)" BUILD_TLS=no

WORKDIR /module
COPY --from=source-verifier /source/lib/kv-router/valkey-module/ .
RUN make VALKEY_SRC=/valkey/src

FROM nvcr.io/nvidian/dynamo-dev/biswa:prime-p4-83a1c54f-d99e6775-dspark-fi0613-vllm2c17d33f4@sha256:f2630341da92fa7841a86d684b9e58c9b940e3193f9c6062571ea45e37d38b07

ARG GIT_REVISION

USER root

ENV DYNAMO_IMAGE_GIT_REVISION=${GIT_REVISION}
ENV VALKEY_IMAGE_GIT_REVISION=5b690cefd6cad707a748879c2bab6b72e18efcb7

COPY --from=dynamo-builder /out/ /tmp/wheels/
COPY --from=valkey-builder /valkey/src/valkey-server /usr/local/bin/valkey-server
COPY --from=valkey-builder /valkey/src/valkey-cli /usr/local/bin/valkey-cli
COPY --from=valkey-builder /module/dynkv.so /usr/local/lib/dynkv.so
COPY --from=source-verifier /source/benchmarks/router/kubernetes/valkey_sweep/start_valkey.py /usr/local/libexec/dynamo/start_valkey.py
COPY --from=source-verifier /source/benchmarks/router/kubernetes/valkey_sweep/pylock.aiperf.toml /tmp/pylock.aiperf.toml

RUN /app/.venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
        /tmp/wheels/ai_dynamo_runtime-*.whl /tmp/wheels/ai_dynamo-*.whl \
    && uv venv --python /app/.venv/bin/python /opt/aiperf-venv \
    && uv pip install --python /opt/aiperf-venv/bin/python --no-cache --no-deps \
        /tmp/wheels/crick-*.whl \
    && uv pip install --python /opt/aiperf-venv/bin/python --no-cache --no-build \
        -r /tmp/pylock.aiperf.toml \
    && rm -rf /tmp/wheels \
    && rm -f /tmp/pylock.aiperf.toml \
    && /app/.venv/bin/python -c \
        'import dynamo._core as c, os; assert c.__build_git_revision__ == os.environ["DYNAMO_IMAGE_GIT_REVISION"]; assert c.__build_git_dirty__ is False' \
    && /app/.venv/bin/python -m dynamo.frontend --help >/dev/null \
    && /app/.venv/bin/python -m dynamo.mocker --help >/dev/null \
    && /opt/aiperf-venv/bin/python -c 'from crick import TDigest; TDigest()' \
    && /opt/aiperf-venv/bin/aiperf --version

ENV HF_HOME=/models

LABEL org.opencontainers.image.revision=${GIT_REVISION} \
      org.opencontainers.image.source="ssh://git@localhost:2224/biswa/dynamo2.git" \
      org.opencontainers.image.valkey-revision="5b690cefd6cad707a748879c2bab6b72e18efcb7"

# Runtime code and the launcher are copied only from the Git object database in
# source.bundle after its HEAD, object graph, and clean checkout are verified.

USER 1000:1000
WORKDIR /app
