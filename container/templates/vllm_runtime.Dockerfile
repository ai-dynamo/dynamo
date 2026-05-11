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
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG NIXL_REF
ARG VLLM_OMNI_REF

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
{% if device != "cuda" %}
ENV PATH=/usr/local/ucx/bin:/usr/local/bin/etcd:${PATH}
{% else %}
ENV PATH=/usr/local/bin/etcd:${PATH}
{% endif %}

ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV TORCH_LIB_DIR=${SITE_PACKAGES}/torch/lib

{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${TORCH_LIB_DIR}:\
${LD_LIBRARY_PATH:-}
{% elif device == "cpu" %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${TORCH_LIB_DIR}:\
${LD_LIBRARY_PATH:-}
{% else %}
# Upstream vLLM ships NIXL and its UCX runtime assets inside the Python
# installation rather than under /opt/nvidia/nvda_nixl. Resolve the packaged
# CUDA-matched `.nixl_cu*` directory once at build time and expose it via a
# stable path so the same runtime template works for both CUDA 12 and CUDA 13.
ENV NIXL_PREFIX=/opt/dynamo/nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV CUDA_RUNTIME_LIB_DIR=/opt/dynamo/cuda-runtime
ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
# Upstream vLLM bundles torch/CUDA wheels under site-packages. Keep their
# runtime libs visible after resetting the image entrypoint, or LMCache's
# compiled extension fails to load libc10/libcudart during serve startup.
${TORCH_LIB_DIR}:\
${CUDA_RUNTIME_LIB_DIR}:\
${LD_LIBRARY_PATH:-}
{% endif %}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
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

{% if device == "cuda" %}
# Upstream vLLM v0.19.1 currently ships NIXL 0.9.0, whose wheels omit
# libnixl_capi.so. Upgrade both CUDA wheel variants so nixl_sys stubs and the
# runtime-selected NIXL path see the same C API capable package.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    NIXL_VERSION="${NIXL_REF#v}"; \
    uv pip install --system --force-reinstall --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu12==${NIXL_VERSION}" \
        "nixl-cu13==${NIXL_VERSION}"

# Find upstream's CUDA-versioned NIXL and libcudart libs and expose them at
# stable paths. CUDA 12 and CUDA 13 package the runtime under different
# site-packages directories, so resolve the actual location once at build time.
RUN set -eu; \
    NIXL_SITE_DIR="$(find "${SITE_PACKAGES}" -maxdepth 1 -type d -name ".nixl_cu${CUDA_MAJOR}.mesonpy.libs" | sort | tail -n 1)"; \
    if [ -z "${NIXL_SITE_DIR}" ]; then \
        echo "Could not find CUDA-matched NIXL libs under ${SITE_PACKAGES}" >&2; \
        find "${SITE_PACKAGES}" -maxdepth 1 -type d -name ".nixl_cu${CUDA_MAJOR}.mesonpy.libs" >&2; \
        exit 1; \
    fi; \
    ln -sfn "${NIXL_SITE_DIR}" "${NIXL_PREFIX}"; \
    CUDA_RUNTIME_SITE_LIB="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type f -name 'libcudart.so.*' | sort | tail -n 1)"; \
    test -n "${CUDA_RUNTIME_SITE_LIB}"; \
    ln -sfn "$(dirname "${CUDA_RUNTIME_SITE_LIB}")" "${CUDA_RUNTIME_LIB_DIR}"
{% endif %}

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Upgrade NIXL meta package and all device variants to match our built version.
# The nixl meta package imports device-specific packages, so all must be at the same version.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    NIXL_VERSION="${NIXL_REF#v}"; \
    uv pip install \
{% if device == "cuda" %}
        --system \
{% endif %}
        --force-reinstall --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu12==${NIXL_VERSION}" \
        "nixl-cu13==${NIXL_VERSION}"

# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.
# Install Dynamo wheels and device-specific NIXL (for CPU/XPU).
# Use --no-deps to prevent dependency conflicts (e.g., KVBM downgrading nixl).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install \
{% if device == "cuda" %}
        --system \
{% endif %}
        --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
{% if device != "cuda" %}
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
{% endif %}
    && if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install \
{% if device == "cuda" %}
            --system \
{% endif %}
            --no-deps "$KVBM_WHEEL"; fi; \
    fi{% if device == "cuda" %} && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --system --no-deps "$GMS_WHEEL"; fi; \
    fi{% endif %}

# vLLM-Omni's audio helpers shell out to SoX, and the launch script examples use
# jq for readable curl output just like the upstream omni image does.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        jq \
        sox \
        libsox-fmt-all; \
    rm -rf /var/lib/apt/lists/*

# Layer vLLM-Omni on top of the upstream vLLM runtime without letting pip
# silently replace the base image's torch/vLLM/transformers stack. Keep only
# the protected base-package policy locally, and let a tiny helper script fetch
# the omni extras directly from the pinned upstream ref while restoring the
# upstream `vllm` CLI that vllm-omni overwrites.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    export VLLM_OMNI_TARGET_DEVICE={{ device }}; \
    bash /tmp/install_vllm_omni.sh

{% endif %}

USER dynamo

# Copy the workspace surface needed by the current vLLM pre-merge test image.
# Keep optional framework trees like planner out of /workspace so the upstream
# runtime does not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/vllm /workspace/components/src/dynamo/vllm
COPY --chown=dynamo:0 lib /workspace/lib
COPY --chmod=775 --chown=dynamo:0 deploy/sanity_check.py /workspace/deploy/sanity_check.py

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset the upstream "vllm serve" entrypoint so the derived runtime behaves
# like other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []
