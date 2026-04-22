{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/planner.Dockerfile ===
##############################################
########## Planner / Profiler image ##########
##############################################
# Standalone planner/profiler image:
# - install deps in a slim builder stage that has git/git-lfs available
# - ship only the runtime artifacts in a distroless final stage

FROM ${PLANNER_BUILD_IMAGE}:${PLANNER_BUILD_IMAGE_TAG} AS planner_builder

ARG PYTHON_VERSION

# Install only the packages needed to resolve and install the planner runtime
# dependencies in the builder stage. git/git-lfs are only needed because
# aiconfigurator is currently installed from a Git URL with LFS-backed assets.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create dynamo user with group 0 for OpenShift compatibility.
RUN useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo /workspace \
    && chown -R dynamo:0 /home/dynamo /opt/dynamo /workspace \
    && chmod -R g+w /home/dynamo/.cache /opt/dynamo /workspace

ENV HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:/usr/local/bin/etcd:/usr/local/bin:/bin" \
    PYTHONPATH="/workspace"

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=dynamo_base /usr/local/bin/nats-server /usr/local/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd /usr/local/bin/etcd
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

USER dynamo

RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}

# Install the local wheels and planner/profiler runtime dependencies before the
# repo copies so changes in tests/configs don't invalidate the dependency layer.
RUN --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --requirement /tmp/requirements.planner.txt \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl

# Copy only the subset of the repository needed for planner/profiler service
# startup and the component-local planner-family test suites.
COPY --chmod=664 --chown=dynamo:0 pyproject.toml /workspace/pyproject.toml
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/planner /workspace/components/src/dynamo/planner
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/profiler /workspace/components/src/dynamo/profiler
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/global_planner /workspace/components/src/dynamo/global_planner
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/

FROM ${PLANNER_RUNTIME_IMAGE}:${PLANNER_RUNTIME_IMAGE_TAG} AS planner

COPY --from=planner_builder /etc/group /etc/passwd /etc/
COPY --from=planner_builder /bin/dash /bin/sh
COPY --from=planner_builder /bin/uv /bin/uvx /usr/local/bin/
COPY --chown=1000:0 --from=planner_builder /home/dynamo /home/dynamo
COPY --chown=1000:0 --from=planner_builder /opt/dynamo/venv /opt/dynamo/venv
COPY --from=planner_builder /usr/lib/*-linux-gnu/libgomp.so.1* /opt/dynamo/lib/
COPY --from=planner_builder /usr/local/bin/etcd /usr/local/bin/etcd
COPY --from=planner_builder /usr/local/bin/nats-server /usr/local/bin/nats-server
COPY --chown=1000:0 --from=planner_builder /workspace /workspace

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA} \
    HOME=/home/dynamo \
    VIRTUAL_ENV=/opt/dynamo/venv \
    LD_LIBRARY_PATH="/opt/dynamo/lib" \
    PATH="/opt/dynamo/venv/bin:/usr/local/bin/etcd:/usr/local/bin:/bin" \
    PYTHONPATH="/workspace/components/src:/workspace"

WORKDIR /workspace
USER dynamo

CMD []
