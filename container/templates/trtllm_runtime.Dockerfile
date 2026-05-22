{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

# Workspace transport stage: gather all workspace contents so the runtime
# pulls them in one COPY layer instead of nine. Reduces overlayfs depth on
# top of the 226-layer upstream TRT-LLM base.
FROM scratch AS workspace_files
COPY --chmod=775 tests /tests
COPY --chmod=775 examples /examples
COPY --chmod=775 deploy /deploy
COPY --chmod=775 dev /dev
COPY --chmod=775 components/src/dynamo/common /components/src/dynamo/common
COPY --chmod=775 components/src/dynamo/frontend /components/src/dynamo/frontend
COPY --chmod=775 components/src/dynamo/trtllm /components/src/dynamo/trtllm
COPY --chmod=775 components/src/dynamo/mocker /components/src/dynamo/mocker
COPY --chmod=775 lib /lib

# Transport stage for dynamo_base artifacts — same one-layer trick as
# workspace_files. Each file is placed at its final runtime path so the
# runtime stage pulls all four with a single cross-stage COPY.
# Note: place uv/uvx at /usr/bin (not /bin) — upstream tensorrt-llm/release is
# usrmerged (/bin -> /usr/bin), and a cross-stage COPY of / / chokes on the
# symlink target. The runtime's PATH includes /usr/bin so this is equivalent.
FROM scratch AS dynamo_base_export
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /usr/bin/uv
COPY --from=dynamo_base /bin/uvx /usr/bin/uvx

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG TARGETARCH
ARG DYNAMO_COMMIT_SHA

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
    NIXL_PLUGIN_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/plugins \
    DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Not in upstream nvcr.io/nvidia/tensorrt-llm/release.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-server \
        librdmacm1 \
        rdma-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Sanity check the libnixl/NIXL_PLUGIN_DIR paths exist (otherwise
#    LD_PRELOAD silently logs "cannot be preloaded: ignored" and the hang
#    returns at runtime). LD_PRELOAD entry 1 (libstdc++) is verified by the
#    ln -sf below; entry 2 (libnixl) is verified here.
# 2. Upstream's /etc/shinit_v2 prepends /usr/local/tensorrt/lib (and others)
#    to LD_LIBRARY_PATH only when a shell starts. K8s spawns python3 directly,
#    so register the paths with ldconfig instead.
# 3. Upstream ships /usr/local/bin/etcd as a single binary; remove it so we
#    can install dynamo_base's etcd directory at the same path below.
# 4. Stable arch-independent symlink to the system libstdc++. LD_PRELOAD'd
#    by the ENV above so external PyInstaller-bundled tools (e.g. NVIDIA's
#    `jet` CI runner) that ship an older libstdc++ in their _MEI extraction
#    dir don't shadow upstream's GCC 13.2 libstdc++ when they dlopen our
#    libnixl / nvda_nixl libs. Drop this once jet stops bundling libstdc++.
RUN test -f /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/nixl/libnixl.so && \
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

# Create dynamo user with group 0 for OpenShift compatibility, clear upstream's
# /workspace baggage (README.md, tutorials/, docker-examples/, license.txt —
# pytest collection picks up broken tutorial test files otherwise), and (for
# non-dev targets) create the dynamo venv. Upstream tensorrt-llm/release marks
# system Python as PEP 668 externally-managed, so we install Dynamo wheels into
# a venv with --system-site-packages so upstream's solve stays importable while
# our wheels live in their own namespace.
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

COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
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

# Still root from above; collapse workspace COPY + launch-screen prep into one
# pair of layers and only switch to dynamo at the very end.
COPY --chmod=775 --chown=dynamo:0 --from=workspace_files / /workspace/
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

# Reset upstream TRT-LLM image's entrypoint so derived runtimes behave like
# other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
CMD ["/bin/bash"]
