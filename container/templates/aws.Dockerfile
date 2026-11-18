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

# Temporarily override NIXL to 1.5.2: vLLM 0.33.0 pins nixl==1.5.0, which leaves
# EFA unsolicited write receive enabled (ai-dynamo/nixl#2529).
# Remove once vLLM's kv_connectors pin is nixl>=1.5.2.
{% if framework == "vllm" %}
RUN --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv pip install --system --no-deps \
        nixl==1.5.2 nixl-cu12==1.5.2 nixl-cu13==1.5.2
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
