{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/planner.Dockerfile ===
##############################################
########## Planner / Profiler image ##########
##############################################
# Standalone planner/profiler image:
# - install dependencies in a slim builder stage
# - ship only the runtime artifacts in a distroless final stage

FROM ${PLANNER_BUILD_IMAGE}:${PLANNER_BUILD_IMAGE_TAG} AS planner_builder

ARG PYTHON_VERSION

# AIConfigurator is installed from an immutable Git revision and its core uses
# maturin, so keep the pinned Dynamo Rust toolchain in this builder stage. The
# distroless runtime below receives only the completed virtual environment.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
COPY --from=dynamo_base /usr/local/rustup /usr/local/rustup
COPY --from=dynamo_base /usr/local/cargo /usr/local/cargo

# Install only the packages needed to build and install planner dependencies.
# build-essential + libc6-dev also cover aiperf's `crick` source build on arm64.
# Python headers come from the base image's
# /usr/local/include/python${PYTHON_VERSION} (python:3.X-slim bundles them
# directly — no apt python*-dev needed, and python${PYTHON_VERSION}-dev is
# not available in this base's apt index anyway).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        git-lfs \
        libc6-dev \
        libgomp1 \
        patchelf \
        pkg-config && \
    git lfs install --system && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create dynamo user with group 0 for OpenShift compatibility.
RUN useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo /workspace \
    && chown -R dynamo:0 /home/dynamo /opt/dynamo /workspace "${RUSTUP_HOME}" "${CARGO_HOME}" \
    && chmod -R g+w /home/dynamo/.cache /opt/dynamo /workspace "${RUSTUP_HOME}" "${CARGO_HOME}"

ENV HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:/opt/uv/bin:/usr/local/cargo/bin:/usr/local/bin/etcd:/usr/local/bin:/bin" \
    PYTHONPATH="/workspace"

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:{{ context.dynamo.uv_version }} /uv /uvx /opt/uv/bin/
COPY --from=dynamo_base /usr/local/bin/nats-server /usr/local/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd /usr/local/bin/etcd
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

USER dynamo

RUN --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}

# Install the local wheels and planner/profiler runtime dependencies before the
# repo copies so changes in tests/configs don't invalidate the dependency layer.
# aiperf is required by the thorough profiler path (profiler/utils/aiperf.py).
RUN --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.benchmark.txt,target=/tmp/requirements.benchmark.txt \
    --mount=type=bind,source=./container/deps/overrides.planner.txt,target=/tmp/overrides.planner.txt \
    --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --overrides /tmp/overrides.planner.txt \
        --requirement /tmp/requirements.planner.txt \
        --requirement /tmp/requirements.benchmark.txt \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/aisimulate*.whl

# Copy only the subset of the repository needed for planner/profiler service
# startup and the component-local planner-family test suites. AI Simulate
# runtime code comes from the wheel installed above; copy only its tests.
COPY --chmod=664 --chown=dynamo:0 pyproject.toml /workspace/pyproject.toml
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/planner /workspace/components/src/dynamo/planner
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/profiler /workspace/components/src/dynamo/profiler
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/global_planner /workspace/components/src/dynamo/global_planner
COPY --chmod=775 --chown=dynamo:0 aisimulate/tests /workspace/aisimulate/tests
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/


{% include "templates/compliance.Dockerfile" %}


FROM ${PLANNER_RUNTIME_IMAGE}:${PLANNER_RUNTIME_IMAGE_TAG} AS planner

COPY --from=planner_builder /etc/group /etc/passwd /etc/
COPY --from=planner_builder /bin/dash /bin/sh
COPY --from=planner_builder /opt/uv/bin/uv /opt/uv/bin/uvx /opt/uv/bin/
COPY --chown=1000:0 --from=planner_builder /home/dynamo /home/dynamo
COPY --chown=1000:0 --from=planner_builder /opt/dynamo/venv /opt/dynamo/venv
COPY --from=planner_builder /usr/lib/*-linux-gnu/libgomp.so.1* /opt/dynamo/lib/
COPY --from=planner_builder /usr/local/bin/etcd /usr/local/bin/etcd
COPY --from=planner_builder /usr/local/bin/nats-server /usr/local/bin/nats-server
COPY --chown=1000:0 --from=planner_builder /workspace /workspace
COPY --from=licenses /legal /legal

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA} \
    HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    LD_LIBRARY_PATH="/opt/dynamo/lib" \
    PATH="/opt/dynamo/venv/bin:/opt/uv/bin:/usr/local/bin/etcd:/usr/local/bin:/bin" \
    PYTHONPATH="/workspace/components/src:/workspace"

WORKDIR /workspace
USER dynamo

CMD []
