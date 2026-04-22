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

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo
ENV PATH=/usr/local/bin/etcd:${PATH}

# Upstream vLLM ships NIXL and its UCX runtime assets inside the Python
# installation rather than under /opt/nvidia/nvda_nixl. Resolve the packaged
# `.nixl_cu*` directory once at build time and expose it via a stable path so
# the same runtime template works for both CUDA 12 and CUDA 13 images.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV NIXL_PREFIX=/opt/dynamo/nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV TORCH_LIB_DIR=${SITE_PACKAGES}/torch/lib
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

# Find upstream's CUDA-versioned NIXL and libcudart libs and expose them at
# stable paths. CUDA 12 and CUDA 13 package the runtime under different
# site-packages directories, so resolve the actual location once at build time.
RUN set -eu; \
    NIXL_SITE_DIR="$(find "${SITE_PACKAGES}" -maxdepth 1 -type d -name '.nixl_cu*.mesonpy.libs' | sort | tail -n 1)"; \
    test -n "${NIXL_SITE_DIR}"; \
    ln -sfn "${NIXL_SITE_DIR}" "${NIXL_PREFIX}"; \
    CUDA_RUNTIME_SITE_LIB="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type f -name 'libcudart.so.*' | sort | tail -n 1)"; \
    test -n "${CUDA_RUNTIME_SITE_LIB}"; \
    ln -sfn "$(dirname "${CUDA_RUNTIME_SITE_LIB}")" "${CUDA_RUNTIME_LIB_DIR}"

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install --system --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --system --no-deps "$GMS_WHEEL"; fi; \
    fi

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
