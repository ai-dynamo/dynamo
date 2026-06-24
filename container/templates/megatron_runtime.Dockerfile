{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/megatron_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

# NGC PyTorch base + Megatron-LM (cloned at build time)

# Transport stage — runtime pulls /workspace_src in one bind-mount cp.
FROM scratch AS workspace_files
COPY --chmod=775 tests /workspace_src/tests
COPY --chmod=775 examples /workspace_src/examples
COPY --chmod=775 deploy /workspace_src/deploy
COPY --chmod=775 dev /workspace_src/dev
COPY --chmod=775 components/src/dynamo/common /workspace_src/components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /workspace_src/components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/megatron /workspace_src/components/src/dynamo/megatron
COPY --chmod=775 lib /workspace_src/lib
COPY --chmod=664 ATTRIBUTION* LICENSE /workspace_src/

# Transport stage for dynamo_base artifacts.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /usr/bin/uv
COPY --from=dynamo_base /bin/uvx /usr/bin/uvx

{% if target == "runtime" %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime_full
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

ARG MEGATRON_REPO
ARG MEGATRON_REF
ARG TARGETARCH

# DYNAMO_HOME points at /workspace. Dynamo + Megatron-LM both install into
# /opt/dynamo/venv (created below with --system-site-packages so upstream
# PyTorch/CUDA libs stay importable)
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    MEGATRON_HOME=/opt/megatron-lm \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin/etcd:${PATH} \
    PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-}

WORKDIR /workspace

# OS packages: git for the Megatron clone, openssh + rdma for downstream
# multi-node bring-up later, python3-venv so we can build /opt/dynamo/venv.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        openssh-server \
        librdmacm1 \
        rdma-core \
        python3-venv

# Pull nats-server, etcd, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /

# Create the Dynamo venv with --system-site-packages so the upstream PyTorch
# image's pinned torch/numpy/triton/cuda libs remain importable. Everything
# below installs into this venv via uv pip (no --system flag).
RUN mkdir -p /opt/dynamo \
    && python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /usr/bin/uv /opt/dynamo/venv/bin/uv

# Clone Megatron-LM at the configured ref and install it editable so users
# can later mount their own checkout over /opt/megatron-lm at `docker run`
# time without rebuilding. --no-deps preserves the NGC PyTorch image's solve
RUN git clone --depth 1 --branch "${MEGATRON_REF}" "${MEGATRON_REPO}" /opt/megatron-lm \
    && uv pip install --no-deps -e /opt/megatron-lm

# Megatron-Inference + the Dynamo backend's runtime dep set. Kept narrow
# (msgpack + pyzmq for the wire protocol, msgspec/uvloop for ai-dynamo-runtime,
# transformers for the tokenizer path)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.megatron.txt,target=/tmp/requirements.megatron.txt \
    uv pip install --no-deps --requirement /tmp/requirements.megatron.txt

# Megatron's `ssm` extra — required by hybrid/SSM models 
# --no-build-isolation builds against the base image's torch/CUDA
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps ninja && \
    MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv pip install --no-deps --no-build-isolation "causal-conv1d~=1.5" "mamba-ssm~=2.2"

# Dynamo user (group 0 for OpenShift), reset upstream /workspace baggage.
# /opt/dynamo was already created above to host the venv. Must come BEFORE
# the NIXL COPYs below — those use --chown=dynamo:0 and would fail with
# "invalid user" if the user doesn't exist yet.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace /opt/megatron-lm \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# NIXL native libs + Python wheels, copied from wheel_builder
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins

ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
/usr/local/cuda/lib64:\
/usr/local/cuda/compat:\
${LD_LIBRARY_PATH:-}

# Native NIXL + UCX runtime libs, and the locally-built wheels.
COPY --chown=dynamo:0 --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo:0 --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Place wheels in /opt/dynamo/wheelhouse unconditionally — dev/local-dev
# images install from source and skip the pip install RUN below but still
# need the wheels on disk.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    # Dynamo wheels --no-deps preserves upstream's solve
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

# Pull /workspace_src (incl. ATTRIBUTION/LICENSE) from the transport stage.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

ENTRYPOINT []
CMD ["/bin/bash"]

{% if target == "runtime" %}
# Rebase on upstream so this stage inherits upstream's image config
# (ENV/WORKDIR/USER/CMD) and overlay runtime_full's filesystem as a single
# layer.
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
RUN rm -rf /workspace /home/ubuntu /usr/local/bin/etcd
COPY --from=runtime_full / /

ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    MEGATRON_HOME=/opt/megatron-lm \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin/etcd:${PATH} \
    PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-} \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib64/plugins:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

WORKDIR /workspace

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

USER dynamo

ENTRYPOINT []
CMD ["/bin/bash"]
{% endif %}
