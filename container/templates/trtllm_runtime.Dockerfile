{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH

WORKDIR /workspace

# DYNAMO_HOME points at /workspace so bundled TRT-LLM scripts that reference
# $DYNAMO_HOME/examples/... resolve. Examples and tests are copied below.
ENV DYNAMO_HOME=/workspace
ENV HOME=/home/dynamo
ENV PATH=/usr/local/bin/etcd:${PATH}

# Upstream's /etc/shinit_v2 prepends /usr/local/tensorrt/lib (and others) to
# LD_LIBRARY_PATH, but only when a shell starts. K8s spawns python3 directly,
# so the libs aren't found and `import tensorrt` fails. Register the paths
# with ldconfig instead so they resolve regardless of launch method.
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    printf '%s\n' \
        "/usr/local/tensorrt/lib" \
        "/usr/local/cuda/lib64" \
        "/usr/local/ucx/lib" \
        "/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu" \
        "/opt/nvidia/nvda_nixl/lib64" \
        > /etc/ld.so.conf.d/00-dynamo-trtllm.conf && \
    ldconfig

# Upstream ships /usr/local/bin/etcd as a single binary; remove it so we can
# install dynamo_base's etcd directory (etcd+etcdctl+etcdutl) at the same path.
RUN rm -f /usr/local/bin/etcd
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
# Copy uv from dynamo_base so the venv symlink below resolves even if a future
# upstream tensorrt-llm/release tag drops /usr/local/bin/uv from its image.
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility.
# Also clear upstream's /workspace baggage (README.md, tutorials/, docker-examples/,
# license.txt) — pytest collection picks up broken tutorial test files otherwise.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Upstream tensorrt-llm/release marks system Python as PEP 668 externally-managed.
# Install Dynamo wheels into a venv with --system-site-packages so upstream's
# solve stays importable while our wheels live in their own namespace.
RUN python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /usr/bin/uv /opt/dynamo/venv/bin/uv
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:${PATH}

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.trtllm.txt,target=/tmp/requirements.trtllm.txt \
    export UV_CACHE_DIR=/root/.cache/uv && \
    \
    # Dynamo's own wheels — --no-deps preserves upstream's solve.
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    \
    # Third-party deps Dynamo wheels declare but upstream lacks, plus the
    # huggingface-hub pin and KVBM-matching nixl-cu13. See the file for context.
    uv pip install --no-deps --requirement /tmp/requirements.trtllm.txt && \
    \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM=true but no kvbm*.whl found in /opt/dynamo/wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install --no-deps "$KVBM_WHEEL"; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --no-deps "$GMS_WHEEL"; fi; \
    fi
{% endif %}

USER dynamo

# Copy the workspace surface needed by trtllm pre-merge tests.
# Keep optional framework trees out of /workspace so the upstream runtime does
# not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/trtllm /workspace/components/src/dynamo/trtllm
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 lib /workspace/lib

RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset upstream TRT-LLM image's entrypoint so derived runtimes behave like
# other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
