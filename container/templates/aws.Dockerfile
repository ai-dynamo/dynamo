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

ARG EFA_VERSION

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

{% if framework == "trtllm" %}
# EFA uses NIXL's LIBFABRIC backend, which the base image's NIXL lacks. Preload
# the PyPI build of the same NIXL version (the base image's NIXL_VERSION, or
# NIXL_EFA_VERSION when set), which has it, against the EFA installer's
# libfabric. libcuda.so.1 comes from the host driver at runtime.
ARG NIXL_EFA_VERSION=""
RUN NIXL_EFA_VERSION="${NIXL_EFA_VERSION:-${NIXL_VERSION:-}}" && \
    [ -n "${NIXL_EFA_VERSION}" ] || { echo "set NIXL_EFA_VERSION: base image does not set NIXL_VERSION" >&2; exit 1; } && \
    /opt/uv/bin/uv pip install --no-deps --target /opt/dynamo/nixl-efa "nixl-cu13==${NIXL_EFA_VERSION}" && \
    NIXL_EFA_LIB_DIR=/opt/dynamo/nixl-efa/.nixl_cu13.mesonpy.libs && \
    test -f "${NIXL_EFA_LIB_DIR}/libnixl.so" && \
    test -f "${NIXL_EFA_LIB_DIR}/plugins/libplugin_LIBFABRIC.so" && \
    if ldd "${NIXL_EFA_LIB_DIR}/plugins/libplugin_LIBFABRIC.so" | grep -v 'libcuda\.so' | grep 'not found'; then \
        echo "libplugin_LIBFABRIC.so has unresolved dependencies" >&2; exit 1; \
    fi

ENV LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/opt/dynamo/nixl-efa/.nixl_cu13.mesonpy.libs/libnixl.so
ENV NIXL_PLUGIN_DIR=/opt/dynamo/nixl-efa/.nixl_cu13.mesonpy.libs/plugins
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
