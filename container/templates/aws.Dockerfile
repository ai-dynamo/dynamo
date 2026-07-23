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

FROM ${EFA_BASE_IMAGE} AS aws_base

ARG EFA_VERSION
ARG EFA_INSTALLER_SHA256

USER root

# Install the pinned EFA 1.49 NGC userspace stack. Its AWS-patched libfabric is
# used unchanged; no source-built libfabric is overlaid afterward.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/efa-installer,sharing=locked \
    --mount=type=bind,source=./container/deps/efa/install_efa.sh,target=/tmp/install_efa.sh,readonly \
    /tmp/install_efa.sh "${EFA_VERSION}" "${EFA_INSTALLER_SHA256}" && \
    # Preserve the existing Dynamo runtime behavior: EFA supplies libfabric,
    # but its NCCL-OFI plugin is disabled because it can crash TensorRT-LLM.
    rm -rf /opt/amazon/aws-ofi-nccl /opt/amazon/ofi-nccl && \
    rm -f /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    printf '%s\n' /opt/amazon/efa/lib > /etc/ld.so.conf.d/000_efa.conf && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

RUN /opt/amazon/efa/bin/fi_info --version | grep -F "libfabric: 2.4.0amzn5.0"

FROM aws_base AS aws

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# The SGLang EFA image replaces the upstream NIXL wheel with a wheel built from
# NIXL 1.3.2. The LIBFABRIC plugin must resolve the EFA installer libfabric,
# not a source-built overlay.
RUN set -eux; \
    python3 -c 'import importlib.metadata as m; assert m.version("nixl") == "1.3.2"; assert m.version("nixl-cu13") == "1.3.2"'; \
    site_packages=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    plugins=$(find "${site_packages}" -path '*nixl*' -name libplugin_LIBFABRIC.so -print); \
    test -n "${plugins}"; \
    mkdir -p /tmp/cuda-stubs; \
    if [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1; \
    fi; \
    plugin_libs="/tmp/cuda-stubs:/opt/amazon/efa/lib:${site_packages}/.nixl_cu13.mesonpy.libs:${site_packages}/nixl_cu13.libs:${site_packages}/nixl_cu13.libs/nixl"; \
    for plugin in ${plugins}; do \
        env -u LD_PRELOAD LD_LIBRARY_PATH="${plugin_libs}" ldd "${plugin}" | tee /tmp/nixl-libfabric.ldd; \
        if grep -Fq "not found" /tmp/nixl-libfabric.ldd; then exit 1; fi; \
        grep -F "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" /tmp/nixl-libfabric.ldd; \
    done
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
# Dynamo-built NIXL 1.0.1 plugins). LIBFABRIC goes through libfabric directly
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
