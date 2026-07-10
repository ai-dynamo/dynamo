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
    rm -rf /opt/amazon/aws-ofi-nccl /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
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
        rm -rf /opt/amazon/efa && \
        cp -Pfr /tmp/libfabric_build /opt/amazon/efa && \
        sed -i 's|^prefix=.*|prefix=/opt/amazon/efa|' /opt/amazon/efa/lib/pkgconfig/libfabric.pc && \
        echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/000_efa.conf && \
        rm -f /etc/ld.so.conf.d/efa.conf && \
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
RUN --mount=from=wheel_builder,source=/opt/nvidia/nvda_nixl,target=/tmp/nvda_nixl \
    rm -rf /opt/nvidia/nvda_nixl && \
    cp -Pfr /tmp/nvda_nixl /opt/nvidia/nvda_nixl && \
    export LD_PRELOAD=/opt/nvidia/nvda_nixl/lib64/libnixl.so && \
    export NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins && \
    ldconfig

ENV LD_PRELOAD=/opt/nvidia/nvda_nixl/lib64/libnixl.so
ENV NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins
{% endif %}

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# Build only the NIXL 1.3.0 LIBFABRIC plugin carrying f29. This intermediate
# stage uses the final EFA/libfabric and SGLang wheel, then is discarded so no
# compiler or build dependency is added to the runtime image.
FROM aws_base AS sglang_nixl_efa_builder

USER root

ARG TARGETARCH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils \
        build-essential \
        ca-certificates \
        git \
        libhwloc-dev \
        libnuma-dev \
        ninja-build \
        patchelf \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    python3 -m pip install --break-system-packages \
        "meson=={{ context.sglang.efa_nixl_patch.meson_version }}"

COPY --chmod=0755 container/deps/sglang/nixl/build_f29_plugin.sh \
    /tmp/nixl-f29/build_f29_plugin.sh
COPY --chmod=0644 container/deps/sglang/nixl/f29b589b.patch \
    /tmp/nixl-f29/f29b589b.patch

RUN TARGETARCH="${TARGETARCH}" \
    SGLANG_EFA_NIXL_VERSION="{{ context.sglang.efa_nixl_patch.nixl_version }}" \
    SGLANG_EFA_NIXL_BASE_REF="{{ context.sglang.efa_nixl_patch.base_ref }}" \
    SGLANG_EFA_NIXL_BASE_TREE="{{ context.sglang.efa_nixl_patch.base_tree }}" \
    SGLANG_EFA_NIXL_BASE_SOURCE_SHA256="{{ context.sglang.efa_nixl_patch.base_source_sha256 }}" \
    SGLANG_EFA_NIXL_PATCH_REF="{{ context.sglang.efa_nixl_patch.patch_ref }}" \
    SGLANG_EFA_NIXL_PATCH_SHA256="{{ context.sglang.efa_nixl_patch.patch_sha256 }}" \
    SGLANG_EFA_NIXL_PATCHED_TREE="{{ context.sglang.efa_nixl_patch.patched_tree }}" \
    SGLANG_EFA_NIXL_PATCHED_SOURCE_SHA256="{{ context.sglang.efa_nixl_patch.patched_source_sha256 }}" \
    SGLANG_EFA_NIXL_MESON_VERSION="{{ context.sglang.efa_nixl_patch.meson_version }}" \
    /tmp/nixl-f29/build_f29_plugin.sh \
        /tmp/nixl-f29/f29b589b.patch \
        /opt/dynamo/nixl-f29
{% endif %}

# Keep aws as the public final-stage name used by the shared image workflow.
FROM aws_base AS aws

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
USER root

ARG TARGETARCH

# The wheel contains the active mesonpy plugin and a compatibility copy. Patch
# both, preserving each stock file's own RPATH/RUNPATH and hashed dependencies,
# so NIXL_PLUGIN_DIR overrides cannot select the unfixed implementation.
COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-f29/plugins/mesonpy/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so
COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-f29/plugins/compat/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so
COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-f29/provenance/ \
    /opt/dynamo/patches/nixl-f29/

# Fail closed if the inherited SGLang image changes its NIXL version/layout or
# if anything other than the two plugin files changed across the stage copy.
RUN EXPECTED_NIXL_VERSION="{{ context.sglang.efa_nixl_patch.nixl_version }}" \
    python3 -c 'import importlib.metadata as m, os, sys; actual = m.version("nixl-cu13"); expected = os.environ["EXPECTED_NIXL_VERSION"]; actual == expected or sys.exit(f"expected nixl-cu13=={expected}, found {actual}")' && \
    test "$(cat /opt/dynamo/patches/nixl-f29/BASE_COMMIT)" = "{{ context.sglang.efa_nixl_patch.base_ref }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/BASE_TREE)" = "{{ context.sglang.efa_nixl_patch.base_tree }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/BASE_SOURCE_FILE_SHA256)" = "{{ context.sglang.efa_nixl_patch.base_source_sha256 }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/SOURCE_COMMIT)" = "{{ context.sglang.efa_nixl_patch.patch_ref }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/PATCHED_TREE)" = "{{ context.sglang.efa_nixl_patch.patched_tree }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/PATCH_SHA256)" = "{{ context.sglang.efa_nixl_patch.patch_sha256 }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/SOURCE_FILE_SHA256)" = "{{ context.sglang.efa_nixl_patch.patched_source_sha256 }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/NIXL_VERSION)" = "{{ context.sglang.efa_nixl_patch.nixl_version }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/MESON_VERSION)" = "{{ context.sglang.efa_nixl_patch.meson_version }}" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/SOURCE_URL)" = "https://github.com/ai-dynamo/nixl" && \
    test "$(cat /opt/dynamo/patches/nixl-f29/TARGETARCH)" = "${TARGETARCH}" && \
    test "$(sha256sum /opt/dynamo/patches/nixl-f29/f29b589b.patch | awk '{print $1}')" = "{{ context.sglang.efa_nixl_patch.patch_sha256 }}" && \
    sha256sum -c /opt/dynamo/patches/nixl-f29/NIXL_CORE_SHA256SUMS && \
    sha256sum -c /opt/dynamo/patches/nixl-f29/SHA256SUMS

LABEL com.nvidia.dynamo.sglang.nixl-efa.base-revision="{{ context.sglang.efa_nixl_patch.base_ref }}" \
      com.nvidia.dynamo.sglang.nixl-efa.patch-revision="{{ context.sglang.efa_nixl_patch.patch_ref }}"
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
