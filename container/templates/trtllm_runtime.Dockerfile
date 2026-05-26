{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

# Transport stage — runtime pulls /workspace_src/ in one bind-mount cp.
FROM scratch AS workspace_files
COPY --chmod=775 tests /workspace_src/tests
COPY --chmod=775 examples /workspace_src/examples
COPY --chmod=775 deploy /workspace_src/deploy
COPY --chmod=775 dev /workspace_src/dev
COPY --chmod=775 components/src/dynamo/common /workspace_src/components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /workspace_src/components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/trtllm /workspace_src/components/src/dynamo/trtllm
COPY --chmod=775 components/src/dynamo/mocker /workspace_src/components/src/dynamo/mocker
COPY --chmod=775 lib /workspace_src/lib
COPY --chmod=664 ATTRIBUTION* LICENSE /workspace_src/

# Transport stage for dynamo_base artifacts. uv/uvx go to /usr/bin (not /bin)
# because upstream is usrmerged and cross-stage COPY chokes on the symlink.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /usr/bin/uv
COPY --from=dynamo_base /bin/uvx /usr/bin/uvx

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH

# DYNAMO_HOME points at /workspace so bundled TRT-LLM scripts that reference
# $DYNAMO_HOME/examples/... resolve. LD_PRELOAD/NIXL_PLUGIN_DIR are a workaround
# for ai-dynamo/nixl#1668: nixl-cu13's bundled UCX 1.20.0 hangs in
# `uct_md_query_tl_resources` (md_resources realloc loop, >1 GiB) when two NIXL
# agents init on the same host. Force-load TRT-LLM's bundled libnixl 0.9.0
# (uses system UCX, no bug). LD_PRELOAD is the only lever: nixl-cu13's
# _bindings.so has DT_RPATH which beats LD_LIBRARY_PATH. Drop the two NIXL
# vars when the upstream issue is fixed.
ENV DYNAMO_HOME=/workspace \
    HOME=/home/dynamo \
    PATH=/usr/local/bin/etcd:${PATH} \
    LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so \
    NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins

WORKDIR /workspace

# Install packages missing from upstream, sanity-check libnixl, register
# TRT-LLM lib paths with ldconfig (upstream's /etc/shinit_v2 only sets them
# for shells, not K8s python3 launches), swap upstream's single-binary etcd
# for dynamo_base's directory, and symlink system libstdc++ to a stable
# path for LD_PRELOAD — keeps PyInstaller-bundled tools (`jet`) from
# shadowing it with an older copy.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-server \
        librdmacm1 \
        rdma-core && \
    test -f /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so && \
    test -d "${NIXL_PLUGIN_DIR}" && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    printf '%s\n' \
        "/usr/local/tensorrt/lib" \
        "/usr/local/cuda/lib64" \
        "/usr/local/ucx/lib" \
        "/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu" \
        "/opt/nvidia/nvda_nixl/lib64" \
        > /etc/ld.so.conf.d/00-dynamo-trtllm.conf && \
    ldconfig && \
    rm -f /usr/local/bin/etcd && \
    mkdir -p /opt/dynamo && \
    ln -sf "/usr/lib/${ARCH_ALT}-linux-gnu/libstdc++.so.6" /opt/dynamo/libstdc++.so.6

# One COPY pulls nats-server, etcd/, uv, uvx into their final paths.
COPY --from=dynamo_base_export / /

# dynamo user (group 0 for OpenShift), clear upstream /workspace baggage
# (otherwise pytest collects broken tutorial test files), and create the
# Dynamo venv on non-dev. --system-site-packages keeps upstream's solve
# importable since system Python is PEP 668 externally-managed.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /workspace && mkdir /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh{% if target not in ("dev", "local-dev") %} \
    && python3 -m venv --system-site-packages /opt/dynamo/venv \
    && ln -sf /usr/bin/uv /opt/dynamo/venv/bin/uv{% endif %}

{% if target not in ("dev", "local-dev") %}
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:${PATH}
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Persist wheels in /opt/dynamo/wheelhouse (tests/dependencies/test_kvbm_imports.py
# greps for them) while installing them in the same RUN — saves the standalone
# COPY layer for *.whl.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=./container/deps/requirements.trtllm.txt,target=/tmp/requirements.trtllm.txt \
    --mount=type=bind,from=wheel_builder,source=/opt/dynamo/dist,target=/tmp/wheels \
    mkdir -p /opt/dynamo/wheelhouse && \
    cp /tmp/wheels/*.whl /opt/dynamo/wheelhouse/ && \
    chown -R dynamo:0 /opt/dynamo/wheelhouse && \
    chmod -R 775 /opt/dynamo/wheelhouse && \
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

# Pull /workspace_src (incl. ATTRIBUTION/LICENSE) from the transport stage and
# wire up the launch screen in a single RUN — saves the standalone workspace COPY layer.
RUN --mount=type=bind,from=workspace_files,source=/workspace_src,target=/tmp/workspace_src \
    --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    cp -a /tmp/workspace_src/. /workspace/ && \
    chown -R dynamo:0 /workspace && \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

# Kept at the bottom — SHA changes per build; layers above stay cached.
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset upstream TRT-LLM image's entrypoint so derived runtimes behave like
# other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
CMD ["/bin/bash"]
