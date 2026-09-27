{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/aws.Dockerfile ===
#############################
########## AWS EFA ##########
#############################
#
# This stage extends the runtime/dev stage with AWS EFA installer
# which includes: libfabric and aws-ofi-nccl plugin
#
# Use this stage when deploying on AWS infrastructure with EFA support

FROM ${EFA_BASE_IMAGE} AS aws

ARG TARGETARCH
ARG EFA_VERSION
ARG CUDA_MAJOR
ARG MOONCAKE_VERSION

{% if target == "runtime" %}
USER root
{% endif %}

# Install AWS EFA installer with bundled libfabric and aws-ofi-nccl
# Flags explanation:
#   --skip-kmod: Skip kernel module installation (handled by host)
#   --skip-limit-conf: Skip ulimit configuration (handled by container runtime)
#   --no-verify: Skip GPG verification (optional, can be removed if verification is needed)
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    cd .. && rm -rf aws-efa-installer* && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"

{% if device == "cuda" and framework in ("vllm", "sglang") %}
{# Install packages the same way each framework runtime template does, so only
   the installer, its cache mount, and cache env differ. Define those as pkg_*
   primitives once here, then reuse them for every package installed below.
   - vLLM   -> uv into system Python, sharing the uv-root cache (vllm_runtime.Dockerfile)
   - SGLang -> plain pip --break-system-packages, pip cache (sglang_runtime.Dockerfile) #}
{% if framework == "vllm" %}
{% set pkg_cache_mount = "--mount=type=cache,id=uv-root-" ~ context.dynamo.uv_version ~ ",target=/root/.cache/uv,sharing=locked" %}
{% set pkg_cache_env = "export UV_CACHE_DIR=/root/.cache/uv" %}
{% set pkg_uninstall = "uv pip uninstall --system" %}
{% set pkg_install = "uv pip install --system --no-deps" %}
{% else %}
{% set pkg_cache_mount = "--mount=type=cache,target=/root/.cache/pip,sharing=locked" %}
{% set pkg_cache_env = "export PIP_CACHE_DIR=/root/.cache/pip" %}
{% set pkg_uninstall = "pip uninstall --break-system-packages -y" %}
{% set pkg_install = "pip install --break-system-packages --no-deps" %}
{% endif %}
# Mooncake EFA wheel swapping
# Mooncake EFA wheels are published for x86_64 only.
RUN {{ pkg_cache_mount }} \
    set -eu; \
    {{ pkg_cache_env }}; \
    if [ "${CUDA_MAJOR}" = "13" ]; then \
        MOONCAKE_PKG=mooncake-transfer-engine-cuda13; \
        MOONCAKE_EFA_PKG=mooncake-transfer-engine-efa-cuda13; \
    else \
        MOONCAKE_PKG=mooncake-transfer-engine; \
        MOONCAKE_EFA_PKG=mooncake-transfer-engine-efa; \
    fi; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "TARGETARCH=${TARGETARCH}: ${MOONCAKE_EFA_PKG} is x86_64-only, \
keeping ${MOONCAKE_PKG} (Mooncake EFA protocol unavailable)"; \
    else \
        {{ pkg_uninstall }} "${MOONCAKE_PKG}" 2>/dev/null || true; \
        {{ pkg_install }} \
            "${MOONCAKE_EFA_PKG}==${MOONCAKE_VERSION}"; \
        # Verify by distribution metadata rather than importing the module: the
        # extension links libcuda.so.1, absent from a GPU-less builder.
        python3 -c "import importlib.metadata as m; m.version('${MOONCAKE_EFA_PKG}')"; \
        ! python3 -c "import importlib.metadata as m; m.version('${MOONCAKE_PKG}')" 2>/dev/null; \
    fi

# Mooncake protocol configuration
# Mooncake EFA wheels are published for x86_64 only,
# Set the mooncake protocol to efa on amd64, other arch to rdma
ARG ARCH_IF_NOT_AMD64=${TARGETARCH#amd64}
ARG PROTOCOL_IF_NOT_AMD64=${ARCH_IF_NOT_AMD64:+rdma}
ENV MOONCAKE_PROTOCOL=${PROTOCOL_IF_NOT_AMD64:-efa}

# upstream vLLM and SGLang images do not have /etc/shinit_v2
# Manually set to ofi so NCCL will find the ofi-nccl transport library
ENV NCCL_NET_PLUGIN=ofi
ENV NCCL_TUNER_PLUGIN=ofi
{% endif %}

{% if framework == "trtllm" %}
# After the upstream mesonpy refactor, libplugin_LIBFABRIC.so lands under the
# Dynamo venv while the rest of the NIXL plugin set (GDS/UCX/POSIX) remains at
# the canonical arch-specific location. Copy LIBFABRIC alongside the others so
# NIXL_PLUGIN_DIR resolves every backend from a single directory, and expose a
# stable arch-agnostic alias at /opt/nvidia/nvda_nixl/plugins.
#
# Also clear LD_PRELOAD (the upstream trtllm_runtime stage's ai-dynamo/nixl#1668
# workaround force-loads TRT-LLM's bundled NIXL 0.9.0; that conflicts with the
# Dynamo-built NIXL 0.10.1 plugins). LIBFABRIC goes through libfabric directly
# (not UCX), so it is unaffected by the UCX 1.20.0 hang that LD_PRELOAD works
# around — and LIBFABRIC is the recommended backend for EFA.
RUN --mount=from=wheel_builder,source=/opt/nvidia/nvda_nixl,target=/tmp/nvda_nixl \
    rm -rf /opt/nvidia/nvda_nixl && \
    cp -Pfr /tmp/nvda_nixl /opt/nvidia/nvda_nixl && \
    export LD_PRELOAD=/opt/nvidia/nvda_nixl/lib64/libnixl.so && \
    export NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins && \
    ldconfig

ENV LD_PRELOAD=/opt/nvidia/nvda_nixl/lib64/libnixl.so
ENV NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
