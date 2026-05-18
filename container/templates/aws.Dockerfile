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
    mkdir -p /tmp/efa && \
    cd /tmp/efa && \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /tmp/efa && \
    # Disable the EFA installer's aws-ofi-nccl plugin: it crashes TRT-LLM at engine init.
    # The plugin is installed at /opt/amazon/ofi-nccl (no `aws-` prefix), but ld.so picks
    # it up via /etc/ld.so.conf.d/aws-ofi-nccl.conf (which DOES carry the `aws-` prefix).
    # Remove both, and also the cuda-dl-base location /opt/amazon/aws-ofi-nccl if present,
    # before re-running ldconfig.
    rm -rf /opt/amazon/aws-ofi-nccl /opt/amazon/ofi-nccl \
           /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"

{% if target == "runtime" %}
USER dynamo
{% endif %}

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
