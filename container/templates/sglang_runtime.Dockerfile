{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if device == "xpu" %}
FROM framework AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

{% if context.sglang.enable_media_ffmpeg == "true" %}
# Copy ffmpeg
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Runtime target installs the prebuilt Dynamo wheels. Dev/local-dev build from
# source later in the shared dev stage after the workspace is bind-mounted.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl

{% if device != "xpu" %}
# Install gpu_memory_service wheel if enabled (CUDA only)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then pip install --no-cache-dir --break-system-packages "$GMS_WHEEL"; fi; \
    fi
{% endif %}
{% endif %}

# Copy tests, deploy and components for CI with correct ownership
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/sglang /workspace/components/src/dynamo/sglang
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc && \
{% if device == "xpu" %}
    echo 'source /opt/intel/oneapi/setvars.sh --force' >> /etc/bash.bashrc && \
    echo 'export LD_LIBRARY_PATH=/opt/dynamo/venv/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
{% endif %}
    mkdir -p /sgl-workspace && \
    ln -sf /workspace /sgl-workspace/dynamo

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

{% if device == "xpu" %}
CMD ["bash", "-c", "source /etc/bash.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
