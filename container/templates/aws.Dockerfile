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
#   --build-ngc turns on --yes, --skip-kmod, --skip-limit-conf and --no-verify (Ubuntu only)
#   --skip-rdma-core skips rdma-core packages
#   --skip-mpi skips mpi packages
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc && \
    cd .. && rm -rf aws-efa-installer* && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"

{% if target == "runtime" %}
USER dynamo
{% endif %}
