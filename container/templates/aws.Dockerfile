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

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
FROM ${EFA_BASE_IMAGE} AS aws_base
{% else %}
FROM ${EFA_BASE_IMAGE} AS aws
{% endif %}

ARG EFA_VERSION
{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
ARG EFA_INSTALLER_SHA256
ARG EFA_INSTALLER_SIZE
ARG TARGETARCH
{% endif %}

{% if target == "runtime" %}
USER root
{% endif %}

# Install AWS EFA installer with bundled libfabric and aws-ofi-nccl
# Flags explanation:
#   --skip-kmod: Skip kernel module installation (handled by host)
#   --skip-limit-conf: Skip ulimit configuration (handled by container runtime)
#   --no-verify: Skip GPG verification (optional, can be removed if verification is needed)
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# Release/1.3 verifies the pinned EFA archive's size and SHA-256 before extraction.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    mkdir -p /tmp/efa && \
    cd /tmp/efa && \
    curl --retry 3 --retry-delay 2 -fsSL -o aws-efa-installer-${EFA_VERSION}.tar.gz \
        https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz && \
    test "$(wc -c < aws-efa-installer-${EFA_VERSION}.tar.gz)" -eq "${EFA_INSTALLER_SIZE}" && \
    printf '%s  %s\n' "${EFA_INSTALLER_SHA256}" \
        "aws-efa-installer-${EFA_VERSION}.tar.gz" | sha256sum -c - && \
    tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    if dpkg-query -W libnccl-ofi >/dev/null 2>&1; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y libnccl-ofi; \
    fi && \
    rm -rf /opt/amazon/efa && \
    ./efa_installer.sh -y --build-ngc --skip-kmod --skip-limit-conf \
        --skip-mpi --skip-plugin --no-verify && \
    if dpkg-query -W libnccl-ofi >/dev/null 2>&1; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y libnccl-ofi; \
    fi && \
    rm -rf /tmp/efa && \
    rm -rf /opt/amazon/ofi-nccl /opt/amazon/aws-ofi-nccl \
        /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    ldconfig
{% else %}
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
{% endif %}

ENV EFA_VERSION="${EFA_VERSION}"

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# Fail closed unless the complete native EFA 1.49 userspace closure is active.
# The upstream libfabric 2.5.1 overlay is intentionally not used for this target.
RUN set -eux; \
    test "${EFA_VERSION}" = "1.49.0"; \
    test "$(dpkg-query -W -f='${Version}' libfabric1-aws)" = "2.4.0amzn5.0"; \
    test "$(dpkg-query -W -f='${Version}' libfabric-aws-bin)" = "2.4.0amzn5.0"; \
    test "$(dpkg-query -W -f='${Version}' libfabric-aws-dev)" = "2.4.0amzn5.0"; \
    dpkg-query -W -f='${Version}\n' rdma-core | grep -E '^63([.]|-)'; \
    dpkg-query -W -f='${Version}\n' libibverbs1 | grep -E '^63([.]|-)'; \
    dpkg-query -W -f='${Version}\n' libibverbs-dev | grep -E '^63([.]|-)'; \
    dpkg-query -W -f='${Version}\n' ibverbs-providers | grep -E '^63([.]|-)'; \
    dpkg-query -W -f='${Version}\n' librdmacm1 | grep -E '^63([.]|-)'; \
    dpkg-query -W -f='${Version}\n' librdmacm-dev | grep -E '^63([.]|-)'; \
    ! dpkg-query -W 'libnccl-ofi*' >/dev/null 2>&1; \
    test ! -e /opt/amazon/ofi-nccl; \
    test "$(readlink /opt/amazon/efa/lib/libfabric.so.1)" = "libfabric.so.1.30.0"; \
    test ! -e /opt/amazon/efa/lib/libfabric.so.1.31.1; \
    REAL=/opt/amazon/efa/lib/libfabric.so.1.30.0; \
    case "${TARGETARCH}" in \
        amd64) \
            MULTIARCH=x86_64-linux-gnu; \
            LIBFABRIC_SHA={{ context.sglang[device_key].efa_runtime_sha256.amd64.libfabric }}; \
            LIBEFA_SHA={{ context.sglang[device_key].efa_runtime_sha256.amd64.libefa }}; \
            LIBIBVERBS_SHA={{ context.sglang[device_key].efa_runtime_sha256.amd64.libibverbs }} ;; \
        arm64) \
            MULTIARCH=aarch64-linux-gnu; \
            LIBFABRIC_SHA={{ context.sglang[device_key].efa_runtime_sha256.arm64.libfabric }}; \
            LIBEFA_SHA={{ context.sglang[device_key].efa_runtime_sha256.arm64.libefa }}; \
            LIBIBVERBS_SHA={{ context.sglang[device_key].efa_runtime_sha256.arm64.libibverbs }} ;; \
        *) exit 1 ;; \
    esac; \
    RUNTIME_LIB=/usr/lib/${MULTIARCH}; \
    test "$(readlink "${RUNTIME_LIB}/libefa.so.1")" = "libefa.so.1.5.63.0"; \
    test "$(readlink "${RUNTIME_LIB}/libibverbs.so.1")" = "libibverbs.so.1.16.63.0"; \
    PROVIDER=${RUNTIME_LIB}/libibverbs/libefa-rdmav59.so; \
    test "$(readlink "$PROVIDER")" = "../libefa.so.1.5.63.0"; \
    printf '%s  %s\n' "$LIBFABRIC_SHA" "$REAL" | sha256sum -c -; \
    printf '%s  %s\n' "$LIBEFA_SHA" "${RUNTIME_LIB}/libefa.so.1" | sha256sum -c -; \
    printf '%s  %s\n' "$LIBIBVERBS_SHA" "${RUNTIME_LIB}/libibverbs.so.1" | sha256sum -c -; \
    printf '%s  %s\n' "$LIBEFA_SHA" "$PROVIDER" | sha256sum -c -; \
    nm -D --undefined-only "$REAL" | grep -w efadv_query_qp_wqs; \
    nm -D --undefined-only "$REAL" | grep -w efadv_query_cq; \
    strings "$REAL" | grep -F efa_data_path_direct_post_read; \
    strings "$REAL" | grep -F efa_data_path_direct_post_write; \
    ldd "$REAL" | tee /tmp/efa-libfabric.ldd; \
    ! grep -Fq "not found" /tmp/efa-libfabric.ldd; \
    LIBEFA_LOADED=$(awk '$1 == "libefa.so.1" {print $3}' /tmp/efa-libfabric.ldd); \
    LIBIBVERBS_LOADED=$(awk '$1 == "libibverbs.so.1" {print $3}' /tmp/efa-libfabric.ldd); \
    test "$(readlink -f "$LIBEFA_LOADED")" = \
        "$(readlink -f "${RUNTIME_LIB}/libefa.so.1.5.63.0")"; \
    test "$(readlink -f "$LIBIBVERBS_LOADED")" = \
        "$(readlink -f "${RUNTIME_LIB}/libibverbs.so.1.16.63.0")"; \
    ldd "$PROVIDER" | tee /tmp/efa-provider.ldd; \
    ! grep -Fq "not found" /tmp/efa-provider.ldd; \
    /opt/amazon/efa/bin/fi_info --version | tee /tmp/efa-fi-info.version; \
    grep -F "libfabric: 2.4.0amzn5.0" /tmp/efa-fi-info.version; \
    printf '%s\n' /opt/amazon/efa/lib > /etc/ld.so.conf.d/000_efa.conf; \
    ldconfig
{% else %}
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

{% if framework == "sglang" and device == "cuda" and target == "runtime" %}
# Build only the NIXL 1.3.0 LIBFABRIC plugin carrying the corrected PR #1966
# semantic backport. This stage uses the final EFA 1.49 headers and libraries,
# then is discarded so no compiler or build dependency reaches the runtime.
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

COPY --chmod=0755 container/deps/sglang/nixl/build_pr1966_plugin.sh \
    /tmp/nixl-pr1966/build_pr1966_plugin.sh
COPY --chmod=0644 container/deps/sglang/nixl/pr1966-1.3.0-backport.patch \
    /tmp/nixl-pr1966/pr1966-1.3.0-backport.patch
COPY --chmod=0755 container/deps/sglang/nixl/validate_pr1966_semantics.py \
    /tmp/nixl-pr1966/validate_pr1966_semantics.py

RUN TARGETARCH="${TARGETARCH}" \
    SGLANG_EFA_NIXL_VERSION="{{ context.sglang.efa_nixl_patch.nixl_version }}" \
    SGLANG_EFA_NIXL_WHEEL_BUILD_REF="{{ context.sglang.efa_nixl_patch.wheel_build_ref }}" \
    SGLANG_EFA_NIXL_WHEEL_BUILD_TREE="{{ context.sglang.efa_nixl_patch.wheel_build_tree }}" \
    SGLANG_EFA_NIXL_BASE_REF="{{ context.sglang.efa_nixl_patch.base_ref }}" \
    SGLANG_EFA_NIXL_BASE_TREE="{{ context.sglang.efa_nixl_patch.base_tree }}" \
    SGLANG_EFA_NIXL_UPSTREAM_PR_HEAD="{{ context.sglang.efa_nixl_patch.upstream_pr_head }}" \
    SGLANG_EFA_NIXL_UPSTREAM_PR_TREE="{{ context.sglang.efa_nixl_patch.upstream_pr_tree }}" \
    SGLANG_EFA_NIXL_PATCH_REF="{{ context.sglang.efa_nixl_patch.backport_ref }}" \
    SGLANG_EFA_NIXL_PATCH_SHA256="{{ context.sglang.efa_nixl_patch.backport_sha256 }}" \
    SGLANG_EFA_NIXL_PATCH_ID="{{ context.sglang.efa_nixl_patch.backport_patch_id }}" \
    SGLANG_EFA_NIXL_PATCHED_TREE="{{ context.sglang.efa_nixl_patch.patched_tree }}" \
    SGLANG_EFA_NIXL_BUILDER_SHA256="{{ context.sglang.efa_nixl_patch.builder_sha256 }}" \
    SGLANG_EFA_NIXL_VALIDATOR_SHA256="{{ context.sglang.efa_nixl_patch.validator_sha256 }}" \
    SGLANG_EFA_NIXL_MESON_VERSION="{{ context.sglang.efa_nixl_patch.meson_version }}" \
    /tmp/nixl-pr1966/build_pr1966_plugin.sh \
        /tmp/nixl-pr1966/pr1966-1.3.0-backport.patch \
        /opt/dynamo/nixl-pr1966 \
        /tmp/nixl-pr1966/validate_pr1966_semantics.py

# Keep aws as the public final-stage name used by the shared image workflow.
FROM aws_base AS aws

USER root

ARG TARGETARCH

COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-pr1966/plugins/mesonpy/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so
COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-pr1966/plugins/compat/libplugin_LIBFABRIC.so \
    /usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so
COPY --from=sglang_nixl_efa_builder \
    /opt/dynamo/nixl-pr1966/provenance/ \
    /opt/dynamo/patches/nixl-pr1966/

# Validate provenance, unchanged NIXL core libraries, both plugin copies, and
# the final image's exact libfabric ABI before publishing either architecture.
RUN set -eux; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/WHEEL_BUILD_COMMIT)" = \
        "{{ context.sglang.efa_nixl_patch.wheel_build_ref }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/WHEEL_BUILD_TREE)" = \
        "{{ context.sglang.efa_nixl_patch.wheel_build_tree }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/BASE_COMMIT)" = \
        "{{ context.sglang.efa_nixl_patch.base_ref }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/BASE_TREE)" = \
        "{{ context.sglang.efa_nixl_patch.base_tree }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/UPSTREAM_PR_HEAD)" = \
        "{{ context.sglang.efa_nixl_patch.upstream_pr_head }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/UPSTREAM_PR_TREE)" = \
        "{{ context.sglang.efa_nixl_patch.upstream_pr_tree }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/BACKPORT_REF)" = \
        "{{ context.sglang.efa_nixl_patch.backport_ref }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/PATCHED_TREE)" = \
        "{{ context.sglang.efa_nixl_patch.patched_tree }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/PATCH_SHA256)" = \
        "{{ context.sglang.efa_nixl_patch.backport_sha256 }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/PATCH_ID)" = \
        "{{ context.sglang.efa_nixl_patch.backport_patch_id }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/BUILDER_SHA256)" = \
        "{{ context.sglang.efa_nixl_patch.builder_sha256 }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/VALIDATOR_SHA256)" = \
        "{{ context.sglang.efa_nixl_patch.validator_sha256 }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/NIXL_VERSION)" = \
        "{{ context.sglang.efa_nixl_patch.nixl_version }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/MESON_VERSION)" = \
        "{{ context.sglang.efa_nixl_patch.meson_version }}"; \
    test "$(cat /opt/dynamo/patches/nixl-pr1966/TARGETARCH)" = "${TARGETARCH}"; \
    printf '%s  %s\n' \
        "{{ context.sglang.efa_nixl_patch.backport_sha256 }}" \
        /opt/dynamo/patches/nixl-pr1966/pr1966-1.3.0-backport.patch | sha256sum -c -; \
    printf '%s  %s\n' \
        "{{ context.sglang.efa_nixl_patch.builder_sha256 }}" \
        /opt/dynamo/patches/nixl-pr1966/build_pr1966_plugin.sh | sha256sum -c -; \
    printf '%s  %s\n' \
        "{{ context.sglang.efa_nixl_patch.validator_sha256 }}" \
        /opt/dynamo/patches/nixl-pr1966/validate_pr1966_semantics.py | sha256sum -c -; \
    sha256sum -c /opt/dynamo/patches/nixl-pr1966/NIXL_CORE_SHA256SUMS; \
    sha256sum -c /opt/dynamo/patches/nixl-pr1966/SHA256SUMS; \
    python3 -c 'import importlib.metadata as m; assert m.version("sglang") == "0.5.14"; assert m.version("nixl") == "1.3.0"; assert m.version("nixl-cu13") == "1.3.0"'; \
    ACTIVE=/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so; \
    COMPAT=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so; \
    CORE=/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs; \
    VENDOR=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs; \
    mkdir -p /tmp/cuda-stubs; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1; \
    for plugin in "$ACTIVE" "$COMPAT"; do \
        ! readelf --wide -S "$plugin" | grep -F .nv_fatbin >/dev/null; \
        ! readelf --version-info "$plugin" | grep -F FABRIC_1.9 >/dev/null; \
        for symbol in fi_getinfo fi_freeinfo fi_dupinfo; do \
            readelf --wide -Ws "$plugin" | \
                grep -E "[[:space:]]${symbol}@FABRIC_1[.]8([[:space:]]|$)"; \
        done; \
        env -u LD_PRELOAD \
            LD_LIBRARY_PATH="/tmp/cuda-stubs:/opt/amazon/efa/lib:${CORE}:${VENDOR}" \
            ldd -v "$plugin" | tee /tmp/nixl-plugin.ldd; \
        ! grep -Fq "not found" /tmp/nixl-plugin.ldd; \
        grep -F "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" \
            /tmp/nixl-plugin.ldd; \
        env -u LD_PRELOAD \
            LD_LIBRARY_PATH="/tmp/cuda-stubs:/opt/amazon/efa/lib:${CORE}:${VENDOR}" \
            PLUGIN="$plugin" python3 -c \
                'import ctypes, os; ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)'; \
    done

ENV LD_PRELOAD=/opt/amazon/efa/lib/libfabric.so.1
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

LABEL com.nvidia.dynamo.sglang.nixl-efa.base-revision="{{ context.sglang.efa_nixl_patch.base_ref }}" \
      com.nvidia.dynamo.sglang.nixl-efa.wheel-revision="{{ context.sglang.efa_nixl_patch.wheel_build_ref }}" \
      com.nvidia.dynamo.sglang.nixl-efa.upstream-pr-head="{{ context.sglang.efa_nixl_patch.upstream_pr_head }}" \
      com.nvidia.dynamo.sglang.nixl-efa.backport-tree="{{ context.sglang.efa_nixl_patch.patched_tree }}"
{% endif %}

{% if target == "runtime" %}
USER dynamo
{% endif %}
