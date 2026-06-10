{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if platform == "multi" %}
FROM --platform=linux/amd64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_amd64
FROM --platform=linux/arm64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS vllm_runtime_arm64
FROM vllm_runtime_${TARGETARCH} AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG VLLM_OMNI_REF
ARG NIXL_REF
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}
ARG MODELEXPRESS_VERSION

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
{% if device != "cuda" %}
ENV PATH=/usr/local/ucx/bin:/usr/local/bin/etcd:${PATH}
{% else %}
ENV PATH=/usr/local/bin/etcd:${PATH}
{% endif %}

{% if device != "cuda" %}
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV TORCH_LIB_DIR=${SITE_PACKAGES}/torch/lib
{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
{% elif device == "cpu" %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
{% endif %}
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
{% else %}
# Expose libnixl.so from the upstream nixl-cu${CUDA_MAJOR} PyPI wheel through a
# stable prefix so non-Python consumers use the same NIXL copy that Python imports.
# This keeps Rust nixl-sys dlopen("libnixl.so") from falling into stub mode in
# processes that do not import the nixl Python package first.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV NIXL_PREFIX=/opt/dynamo/nixl \
    NIXL_LIB_DIR=/opt/dynamo/nixl \
    NIXL_PLUGIN_DIR=/opt/dynamo/nixl/plugins
COPY --chmod=755 container/deps/vllm/install_nixl_from_wheel.sh /usr/local/bin/install_nixl_from_wheel
RUN install_nixl_from_wheel \
    --cuda-major "${CUDA_MAJOR}" \
    --site-packages "${SITE_PACKAGES}" \
    --prefix "${NIXL_PREFIX}" \
    --skip-headers
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:${LD_LIBRARY_PATH:-}
{% endif %}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility.
# Pin -u 1000 explicitly: the vllm/vllm-openai >=0.22 image ships a `vllm` user at
# UID 2000, so after freeing 1000 (ubuntu) useradd would otherwise auto-assign the
# next-highest UID (2001) and fail the `id -u dynamo` == 1000 assertion below.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -u 1000 -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

{% if device != "cuda" %}
# Copy UCX and NIXL from wheel_builder for CPU/XPU devices
# (CUDA devices use NIXL from upstream vLLM wheels)
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder ${NIXL_PREFIX} ${NIXL_PREFIX}
{% if device == "xpu" %}
# XPU NIXL uses lib/x86_64-linux-gnu; copy to NIXL_LIB_DIR to ensure lib dir is populated
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/
{% endif %}
# Copy NIXL Python wheels
COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Install RDMA libraries required for UCX to find RDMA devices
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libibverbs1 \
        rdma-core \
        ibverbs-utils \
        libibumad3 \
        libnuma1 \
        librdmacm1 \
        ibverbs-providers && \
    rm -rf /var/lib/apt/lists/*
{% endif %}

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% set pip_target = "--system" if device == "cuda" else "--python /opt/venv/bin/python" %}
{% if device != "cuda" %}
# NIXL meta package always tries to find a cuda-backend
# https://github.com/ai-dynamo/nixl/blob/v1.1.0/src/bindings/python/nixl-meta/nixl/__init__.py
#
# We therefore install nixl-cu* packages, and use LD_LIBRARY_PATH settings to point to our installation of nixl
# v1.1.0 nixl-cu13 has in-built RPATH point to conflicting built-in libs with symbols unsupported in non-cuda builds.
# we therefore avoid installing nixl-cu13

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    NIXL_VERSION="${NIXL_REF#v}"; \
    uv pip install \
        {{ pip_target }} --force-reinstall --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu12==${NIXL_VERSION}"
{% endif %}

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

# Install device-specific NIXL wheels for non-CUDA devices.
# These are custom-built in wheel_builder and required for dev builds to link against NIXL libraries.
{% if device != "cuda" %}
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.

# Install Dynamo runtime wheels and optional KVBM/GMS wheels.
# Use --no-deps to prevent dependency conflicts (e.g., KVBM downgrading nixl).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install {{ pip_target }} --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install {{ pip_target }} --no-deps "$GMS_WHEEL"; fi; \
    fi

# vLLM-Omni's audio helpers shell out to SoX, and the launch script examples use
# jq for readable curl output just like the upstream omni image does.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq \
        sox \
        libsox-fmt-all; \
    rm -rf /var/lib/apt/lists/*

# Layer the released vLLM-Omni package matching the pinned upstream ref while
# constraining packages already solved in the upstream vLLM image.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    export VLLM_OMNI_TARGET_DEVICE={{ device }}; \
    bash /tmp/install_vllm_omni.sh

{% if device == "xpu" %}
# Remove conflicting standard triton package for XPU and reinstall triton-xpu
# This must be done after vLLM-Omni installation to ensure no dependencies re-install triton
# Reinstalling triton-xpu ensures the triton namespace is properly configured
RUN uv pip uninstall triton && \
    uv pip install --force-reinstall --no-deps triton-xpu
{% endif %}

{% if context.vllm.enable_modelexpress == "true" %}
# Install only the ModelExpress client package. --no-deps preserves the upstream
# vLLM runtime dependency stack.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv pip install --system --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"
{% endif %}

{% endif %}

{% if context.vllm.enable_media_ffmpeg == "true" %}
# Copy ffmpeg libraries from wheel_builder (requires root, runs before USER dynamo)
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

# Replace the upstream vllm/vllm-openai image's imageio-ffmpeg (which ships
# a GPL-encumbered prebuilt ffmpeg binary) with a source install that leaves
# no binary on disk. vLLM-Omni uses diffusers.export_to_video and doesn't
# invoke imageio-ffmpeg, so no IMAGEIO_FFMPEG_EXE is needed — this is
# purely to clear the GPL binary. The --no-binary directive lives in the
# requirements file itself.
RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install {{ pip_target }} --reinstall-package imageio-ffmpeg --no-deps \
        --requirement /tmp/requirements.vllm.txt

# Remove the vLLM source tree shipped in the base image to avoid pytest
# collection conflicts (duplicate conftest plugin registration) and stale
# tool scripts referencing files not present in Dynamo's build context.
RUN rm -rf /workspace/vllm

USER dynamo

# Copy the workspace surface needed by the current vLLM pre-merge test image.
# Keep optional framework trees like planner out of /workspace so the upstream
# runtime does not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/vllm /workspace/components/src/dynamo/vllm
COPY --chown=dynamo:0 lib /workspace/lib

# Setup launch banner in common directory accessible to all users
USER root
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset the upstream "vllm serve" entrypoint so the derived runtime behaves
# like other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
