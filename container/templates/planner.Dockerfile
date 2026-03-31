{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/planner.Dockerfile ===
##############################################
########## Planner / Profiler image ##########
##############################################
# Standalone slim image for planner + profiler workflows.
# Reuses locally built Dynamo wheels from wheel_builder, but does not inherit
# the heavy runtime image or its GPU-native payloads.

FROM python:${PYTHON_VERSION}-slim AS planner

ARG PYTHON_VERSION
ARG TARGETARCH
ARG NATS_VERSION
ARG ETCD_VERSION

# Minimal runtime packages:
# - git for VCS-based Python requirements
# - git-lfs because the pinned aiconfigurator Git dependency tracks files with LFS
# - libgomp1 for scikit-learn / scipy OpenMP runtime support
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
        libgomp1 \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install NATS and etcd so planner/profiler workflows can bring up the local
# control plane without relying on the heavier runtime image.
RUN wget --tries=3 --waitretry=5 \
        https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${TARGETARCH}.deb && \
    dpkg -i nats-server-${NATS_VERSION}-${TARGETARCH}.deb && \
    rm nats-server-${NATS_VERSION}-${TARGETARCH}.deb && \
    wget --tries=3 --waitretry=5 \
        https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${TARGETARCH}.tar.gz \
        -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz

# Create dynamo user with group 0 for OpenShift compatibility.
RUN useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo /workspace \
    && chown -R dynamo:0 /home/dynamo /opt/dynamo /workspace \
    && chmod -R g+w /home/dynamo/.cache /opt/dynamo /workspace

ENV HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:/usr/local/bin/etcd:${PATH}" \
    PYTHONPATH="/workspace"

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

# Copy only the subset of the repository needed for planner/profiler service
# startup and targeted planner/profiler unit tests.
COPY --chmod=664 --chown=dynamo:0 pyproject.toml /workspace/pyproject.toml
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/planner /workspace/components/src/dynamo/planner
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/profiler /workspace/components/src/dynamo/profiler
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/global_planner /workspace/components/src/dynamo/global_planner
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples

USER dynamo

RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}

# Install local wheels and planner/profiler runtime dependencies in one pass
# so the resolver sees the full constraint set together.
RUN --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.profiler.txt,target=/tmp/requirements.profiler.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --requirement /tmp/requirements.planner.txt \
        --requirement /tmp/requirements.profiler.txt \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

CMD []
