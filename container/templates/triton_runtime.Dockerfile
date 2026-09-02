{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/triton_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################
#
# RUNTIME_IMAGE is the upstream Triton Inference Server release from NGC
# (nvcr.io/nvidia/tritonserver:<tag>-py3). It already ships /opt/tritonserver
# (server binary, backends incl. tensorrt, TensorRT + CUDA runtime libraries) and
# a system Python. The Dynamo wheels are installed on top of it here, so the
# resulting image is "Dynamo runtime + Triton" in a single artifact — matching how
# the vLLM / SGLang / TRT-LLM runtimes are composed. Override RUNTIME_IMAGE_TAG to
# track a different Triton release.

# Transport stage — runtime pulls /workspace_src/ in one bind-mount cp.
FROM scratch AS workspace_files
COPY --chmod=775 tests /workspace_src/tests
COPY --chmod=775 examples /workspace_src/examples
COPY --chmod=775 deploy /workspace_src/deploy
COPY --chmod=775 dev /workspace_src/dev
COPY --chmod=775 components/src/dynamo/common /workspace_src/components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /workspace_src/components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/mocker /workspace_src/components/src/dynamo/mocker
COPY --chmod=775 components/src/dynamo/triton /workspace_src/components/src/dynamo/triton
COPY --chmod=775 lib /workspace_src/lib
COPY --chmod=664 ATTRIBUTION* LICENSE /workspace_src/

# Transport stage for dynamo_base artifacts. uv/uvx go to /usr/bin (not /bin)
# because upstream is usrmerged and cross-stage COPY chokes on the symlink.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /usr/bin/uv
COPY --from=dynamo_base /bin/uvx /usr/bin/uvx

# The runtime image is the upstream Triton release with the Dynamo wheels and
# launch assets overlaid directly on top. Building in place (rather than
# re-FROM-ing upstream and copying the whole filesystem across stages) keeps the
# upstream base layers shared instead of duplicated, which roughly halves the
# final image size.
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH

# DYNAMO_HOME points at /workspace so bundled examples that reference
# $DYNAMO_HOME/examples/... resolve. BACKEND_DIR / the /opt/tritonserver entries on
# PATH and LD_LIBRARY_PATH let the Dynamo Triton Runtime start tritonserver and load
# GPU instance groups (incl. the TensorRT backend) out of the box.
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/opt/tritonserver/bin:/usr/local/bin/etcd:${PATH} \
    BACKEND_DIR=/opt/tritonserver/backends \
    LD_LIBRARY_PATH=/opt/tritonserver/lib:/opt/tritonserver/backends:${LD_LIBRARY_PATH}

# Create the default model repository directory for Triton.
# The Dynamo Triton Runtime will mount model repositories into /models, so this directory must exist at runtime.
RUN mkdir -p /models

WORKDIR $DYNAMO_HOME/components/src/dynamo/triton

# python3-venv is the only package needed: the Triton release image ships Python
# without ensurepip (so `python3 -m venv` below fails without it) but already has
# UCX 1.20.0 (/opt/hpcx/ucx), libibverbs/rdma-core, librdmacm, and libb64. NIXL is
# intentionally absent — KVBM is disabled for this runtime because its Python NIXL
# path requires PyTorch, which the Triton py3 image lacks.
#
# ldconfig lets non-login python3 launches resolve libtritonserver.so and the
# backend shared objects; dropping upstream's single-binary etcd lets
# dynamo_base's etcd/ directory take over on PATH.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-venv && \
    test -d /opt/tritonserver/backends && \
    printf '%s\n' /opt/tritonserver/lib /opt/tritonserver/backends \
        > /etc/ld.so.conf.d/00-dynamo-triton.conf && \
    ldconfig && \
    rm -f /usr/local/bin/etcd

# One COPY pulls nats-server, etcd/, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /

# dynamo user (group 0 for OpenShift), clear upstream /workspace baggage, and
# create the Dynamo venv. --system-site-packages keeps the upstream Triton Python
# packages (tritonserver, etc.) importable since system Python is PEP 668
# externally-managed. The Triton release image ships a `triton-server` user at
# UID 1000, so it is removed first to free 1000 for `dynamo` (which other Dynamo
# runtimes and the OpenShift group-0 convention assume). The `cd /workspace`
# after recreating it is required: this RUN's CWD is WORKDIR /workspace, and
# `rm -rf /workspace` leaves the shell on a deleted inode, which makes the
# subsequent `python3 -m venv` abort at startup (getcwd fails: "error evaluating
# path"). Re-entering the fresh directory restores a valid CWD.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && userdel -r triton-server > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace && cd /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh \
    && python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /usr/bin/uv /opt/dynamo/venv/bin/uv

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:${PATH}

# Dynamo wheels built from source in wheel_builder — keeps the image in lockstep
# with this repository, like the other backend runtimes.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    \
    # Dynamo's own wheels — with deps (unlike other backends) so the Python
    # packages the Triton base lacks come from PyPI.
    uv pip install \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    \
    # Triton's Python bindings (from the release image) and the tritonclient gRPC client.
    uv pip install --no-deps /opt/tritonserver/python/tritonserver*.whl && \
    uv pip install tritonclient[grpc]

# Pull /workspace_src (incl. ATTRIBUTION/LICENSE) from the transport stage and
# wire up the launch screen in a single RUN — saves the standalone workspace COPY layer.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/triton.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

# Image config (ENV/WORKDIR are already set above). ENTRYPOINT/CMD override the
# upstream Triton entrypoint so the container drops into a shell, and USER drops
# to the unprivileged dynamo user.
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

USER dynamo

ENTRYPOINT []
CMD ["/bin/bash"]
