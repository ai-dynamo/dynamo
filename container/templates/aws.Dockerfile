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

ARG NIXL_LIBFABRIC_REF

# Copy the wheel_builder-built libfabric and register it with the dynamic linker
# ONLY if the EFA-bundled libfabric is older than NIXL_LIBFABRIC_REF.
# When a future EFA installer ships libfabric >= the version we build, the
# version comparison evaluates to false and this becomes a no-op automatically.
RUN --mount=from=wheel_builder,source=/usr/local/libfabric,target=/tmp/libfabric_build \
    EFA_PC=$(find /opt/amazon/efa -path '*/pkgconfig/libfabric.pc' 2>/dev/null | head -n1) && \
    EFA_LIBFABRIC_RAW=$(cat "$EFA_PC" 2>/dev/null | grep '^Version:' | awk '{print $2}') && \
    EFA_LIBFABRIC_VER=$(echo "$EFA_LIBFABRIC_RAW" | grep -oE '^[0-9]+\.[0-9]+(\.[0-9]+)?') && \
    REF_VER=$(echo "${NIXL_LIBFABRIC_REF}" | sed 's/^v//') && \
    if [ -n "$EFA_LIBFABRIC_VER" ] && [ -n "$REF_VER" ] && \
       [ "$(printf '%s\n' "$EFA_LIBFABRIC_VER" "$REF_VER" | sort -V | head -n1)" = "$EFA_LIBFABRIC_VER" ] && \
       [ "$EFA_LIBFABRIC_VER" != "$REF_VER" ]; then \
        cp -Pf /tmp/libfabric_build/lib/libfabric.so* /opt/amazon/efa/lib/ && \
        if [ -d /opt/amazon/efa/lib64 ]; then \
            cp -Pf /tmp/libfabric_build/lib/libfabric.so* /opt/amazon/efa/lib64/; \
        fi && \
        cp -f /tmp/libfabric_build/bin/fi_info /opt/amazon/efa/bin/fi_info && \
        ldconfig && \
        echo "[aws] libfabric overlay: ${REF_VER} (overwrites EFA stock ${EFA_LIBFABRIC_RAW})"; \
    else \
        echo "[aws] libfabric overlay: skipped (EFA stock ${EFA_LIBFABRIC_RAW:-unknown} >= ${REF_VER})"; \
    fi

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
RUN set -e && \
    arch_libdir=$(find /opt/nvidia/nvda_nixl/lib -maxdepth 1 -type d -name '*-linux-gnu' | head -1) && \
    [ -n "$arch_libdir" ] || { echo "ERROR: no arch-specific NIXL plugin dir under /opt/nvidia/nvda_nixl/lib" >&2; exit 1; } && \
    venv_lf=$(find /opt/dynamo/venv -path '*.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so' | head -1) && \
    [ -n "$venv_lf" ] || { echo "ERROR: no libplugin_LIBFABRIC.so under /opt/dynamo/venv" >&2; exit 1; } && \
    cp -Pf "$venv_lf" "$arch_libdir/plugins/" && \
    ln -sfT "$arch_libdir/plugins" /opt/nvidia/nvda_nixl/plugins && \
    [ -f /opt/nvidia/nvda_nixl/plugins/libplugin_LIBFABRIC.so ] || { echo "ERROR: LIBFABRIC plugin not visible via /opt/nvidia/nvda_nixl/plugins" >&2; ls -la /opt/nvidia/nvda_nixl/plugins/ >&2; exit 1; } && \
    echo "[aws] NIXL plugins consolidated under /opt/nvidia/nvda_nixl/plugins -> $arch_libdir/plugins"

ENV LD_PRELOAD=""
ENV NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/plugins
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
