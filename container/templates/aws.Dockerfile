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
    printf '%s\n' /opt/amazon/efa/lib > /etc/ld.so.conf.d/000_efa.conf && \
    ldconfig

ENV EFA_VERSION="${EFA_VERSION}"
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

RUN /opt/amazon/efa/bin/fi_info --version | grep -F "libfabric: 2.4.0amzn5.0"

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# SGLang 0.5.14 carries NIXL 1.3.0. Rebuild only its LIBFABRIC plugin with the
# release/1.3 backport of NIXL PR #1966, using the final EFA headers and library.
FROM aws_base AS sglang_nixl_efa_builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils \
        build-essential \
        git \
        libhwloc-dev \
        libnuma-dev \
        ninja-build \
        patchelf \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --break-system-packages \
        "meson=={{ context.sglang.efa_nixl_patch.meson_version }}"

COPY --chmod=0755 container/deps/sglang/nixl/build_pr1966_plugin.sh \
    /tmp/build_pr1966_plugin.sh
COPY --chmod=0644 container/deps/sglang/nixl/pr1966-1.3.0-backport.patch \
    /tmp/pr1966-1.3.0-backport.patch

RUN /tmp/build_pr1966_plugin.sh \
    "{{ context.sglang.efa_nixl_patch.nixl_version }}" \
    "{{ context.sglang.efa_nixl_patch.source_ref }}" \
    "{{ context.sglang.efa_nixl_patch.patch_sha256 }}" \
    /tmp/pr1966-1.3.0-backport.patch

FROM aws_base AS aws

COPY --from=sglang_nixl_efa_builder \
    /usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so
COPY --from=sglang_nixl_efa_builder \
    /usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so

# Check the two wheel locations actually used by SGLang. Unset LD_PRELOAD so
# this proves each plugin resolves the EFA library on its own.
RUN set -eux; \
    mkdir -p /tmp/cuda-stubs; \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1; \
    core=/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs; \
    vendor=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs; \
    plugin_libs="/tmp/cuda-stubs:/opt/amazon/efa/lib:${core}:${vendor}"; \
    for plugin in \
        "${core}/plugins/libplugin_LIBFABRIC.so" \
        "${vendor}/nixl/libplugin_LIBFABRIC.so"; do \
        env -u LD_PRELOAD \
            LD_LIBRARY_PATH="${plugin_libs}" \
            ldd "${plugin}" | tee /tmp/nixl-libfabric.ldd; \
        if grep -Fq "not found" /tmp/nixl-libfabric.ldd; then exit 1; fi; \
        grep -F "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" \
            /tmp/nixl-libfabric.ldd; \
        env -u LD_PRELOAD \
            LD_LIBRARY_PATH="${plugin_libs}" \
            PLUGIN="${plugin}" python3 -c \
                'import ctypes, os; ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)'; \
    done

ENV LD_PRELOAD=/opt/amazon/efa/lib/libfabric.so.1
{% else %}
FROM aws_base AS aws
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
