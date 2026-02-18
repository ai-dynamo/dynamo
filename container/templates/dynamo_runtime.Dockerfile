{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dynamo_runtime.Dockerfile ===
#######################################
########## Runtime image ##############
#######################################

FROM dynamo_base AS runtime

ARG ARCH_ALT
ARG PYTHON_VERSION

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

# NIXL environment variables
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins \
    CARGO_TARGET_DIR=/opt/dynamo/target

# Copy ucx and nixl libs
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo: --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo: --from=wheel_builder /opt/nvidia/nvda_nixl/lib64/. ${NIXL_LIB_DIR}/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

{% if context.dynamo.enable_media_ffmpeg == "true" %}
# Copy ffmpeg
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

# Copy built artifacts
COPY --chown=dynamo: --from=wheel_builder $CARGO_TARGET_DIR $CARGO_TARGET_DIR
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

# Install Python for framework=none runtime (cuda-dl-base doesn't include Python)
# This is needed to create venv and install dynamo packages
ARG PYTHON_VERSION
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Switch to dynamo user and create virtual environment
USER dynamo
ENV HOME=/home/dynamo

# Create and activate virtual environment
# Use login shell to pick up umask 002 from /etc/profile.d/00-umask.sh for group-writable files
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]
# Cache uv downloads; uv handles its own locking for the cache.
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Install dynamo wheels (runtime packages only, no test dependencies)
# uv handles its own locking for the cache, no need to add sharing=locked
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "$ENABLE_GPU_MEMORY_SERVICE" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -z "$GMS_WHEEL" ]; then \
            echo "ERROR: ENABLE_GPU_MEMORY_SERVICE is true but no gpu_memory_service wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$GMS_WHEEL"; \
    fi && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM is true but no KVBM wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$KVBM_WHEEL"; \
    fi

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
