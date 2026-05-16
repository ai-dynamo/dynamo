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

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
ENV PATH=/usr/local/bin/etcd:${PATH}

# Upstream ships /usr/local/bin/etcd as a single binary; remove it so we can
# install dynamo_base's etcd directory (etcd+etcdctl+etcdutl) at the same path.
RUN rm -f /usr/local/bin/etcd
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
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
    && ln -sf /usr/local/bin/uv /opt/dynamo/venv/bin/uv
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:${PATH}

# With KVBM, install matching `nixl-cu13==0.10.1` from PyPI so libnixl.so (loaded
# via the wheel's DT_RPATH on `import nixl`) matches KVBM's `nixl-sys=0.10.1`
# ABI. Upstream's NIXL 0.9.0 at /opt/nvidia/nvda_nixl coexists unused because
# Dynamo's --connector kvbm bypasses TRT-LLM's native NIXL transceiver.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    \
    # Dynamo's own wheels — --no-deps preserves upstream's solve.
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    \
    # Deps Dynamo wheels declare but upstream tensorrt-llm/release lacks.
    # (vllm/vllm-openai ships these by default, which is why DYN-2204's vllm
    # path does not need this step.)
    uv pip install --no-deps 'uvloop>=0.21.0' 'msgspec>=0.19.0' && \
    \
    # Lock huggingface-hub to upstream transformers's <1.0 constraint. Without
    # this pin, Dockerfile.test's `uv pip install --requirement requirements.test.txt`
    # later upgrades the venv copy to 1.x (datasets allows it) and breaks
    # `import transformers` against upstream's compiled stack.
    uv pip install --no-deps 'huggingface-hub<1.0,>=0.34.0' && \
    \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then \
            uv pip install --no-deps "$KVBM_WHEEL"; \
            uv pip install --no-deps nixl==0.10.1 nixl-cu13==0.10.1; \
        fi; \
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
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/trtllm /workspace/components/src/dynamo/trtllm
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 lib /workspace/lib
COPY --chmod=775 --chown=dynamo:0 deploy/sanity_check.py /workspace/deploy/sanity_check.py

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
