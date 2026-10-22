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
# Upstream NIXL ships UCX/GDS but no LIBFABRIC backend. Add only the missing
# plugin from the matching release, preserving its native libraries and Python
# bindings. The EFA installer above supplies libfabric at runtime.
ARG NIXL_REF
ARG TARGETARCH
RUN --mount=from=wheel_builder,source=/opt/nvidia/nvda_nixl/lib64/plugins/libplugin_LIBFABRIC.so,target=/tmp/libplugin_LIBFABRIC.so \
    python3 -c 'import importlib.metadata as m, os; assert m.version("nixl") == os.environ["NIXL_REF"].removeprefix("v"), "EFA plugin source must match upstream NIXL"' && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    plugin_dir="/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins" && \
    test -d "${plugin_dir}" && \
    cp /tmp/libplugin_LIBFABRIC.so "${plugin_dir}/" && \
    LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:${LD_LIBRARY_PATH} \
    python3 -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "${plugin_dir}/libplugin_LIBFABRIC.so"
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
