{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/aws.Dockerfile ===
#############################
########## AWS EFA ##########
#############################
#
# This stage extends the runtime/dev stage with the pinned AWS EFA userspace
# stack. NIXL builders and final images consume the same installer libraries.
#
# Use this stage when deploying on AWS infrastructure with EFA support

FROM ${EFA_BASE_IMAGE} AS aws_base

ARG EFA_VERSION
ARG EFA_INSTALLER_SHA256
ARG EFA_INSTALLER_SIZE
ARG TARGETARCH

USER root

# The archive cache is shared with wheel_builder. The helper verifies the
# versioned archive's exact size and SHA-256 before every installation.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/efa-installer,sharing=locked \
    --mount=type=bind,source=./container/deps/efa/install_efa.sh,target=/tmp/install_efa.sh,readonly \
    /tmp/install_efa.sh \
        "${EFA_VERSION}" "${EFA_INSTALLER_SHA256}" "${EFA_INSTALLER_SIZE}"

ENV EFA_VERSION="${EFA_VERSION}"

# Fail closed unless the complete native EFA 1.49 userspace closure is active.
# No source-built upstream libfabric overlay is permitted in an EFA image.
RUN set -eux; \
    test "${EFA_VERSION}" = "{{ context.dynamo.efa_version }}"; \
    test "$(dpkg-query -W -f='${Version}' libfabric1-aws)" = "2.4.0amzn5.0"; \
    test "$(dpkg-query -W -f='${Version}' libfabric-aws-bin)" = "2.4.0amzn5.0"; \
    test "$(dpkg-query -W -f='${Version}' libfabric-aws-dev)" = "2.4.0amzn5.0"; \
    test "$(dpkg-query -W -f='${Version}' rdma-core)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' libibverbs1)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' libibverbs-dev)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' ibverbs-providers)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' librdmacm1)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' librdmacm-dev)" = "63.0-1"; \
    test "$(dpkg-query -W -f='${Version}' libnccl-ofi-ngc-v3)" = "1.20.0-1"; \
    dpkg-query -S /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | \
        grep -F 'libnccl-ofi-ngc-v3:'; \
    test "$(readlink /opt/amazon/efa/lib/libfabric.so.1)" = "libfabric.so.1.30.0"; \
    test ! -e /opt/amazon/efa/lib/libfabric.so.1.31.1; \
    REAL=/opt/amazon/efa/lib/libfabric.so.1.30.0; \
    case "${TARGETARCH}" in \
        amd64) \
            MULTIARCH=x86_64-linux-gnu; \
            LIBFABRIC_SHA={{ context.dynamo.efa_runtime_sha256.amd64.libfabric }}; \
            LIBEFA_SHA={{ context.dynamo.efa_runtime_sha256.amd64.libefa }}; \
            LIBIBVERBS_SHA={{ context.dynamo.efa_runtime_sha256.amd64.libibverbs }} ;; \
        arm64) \
            MULTIARCH=aarch64-linux-gnu; \
            LIBFABRIC_SHA={{ context.dynamo.efa_runtime_sha256.arm64.libfabric }}; \
            LIBEFA_SHA={{ context.dynamo.efa_runtime_sha256.arm64.libefa }}; \
            LIBIBVERBS_SHA={{ context.dynamo.efa_runtime_sha256.arm64.libibverbs }} ;; \
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
    grep -aFq efadv_query_qp_wqs "$REAL"; \
    grep -aFq efadv_query_cq "$REAL"; \
    grep -aFq efa_data_path_direct_post_read "$REAL"; \
    grep -aFq efa_data_path_direct_post_write "$REAL"; \
    ldd "$REAL" | tee /tmp/efa-libfabric.ldd; \
    if grep -Fq "not found" /tmp/efa-libfabric.ldd; then exit 1; fi; \
    LIBEFA_LOADED=$(awk '$1 == "libefa.so.1" {print $3}' /tmp/efa-libfabric.ldd); \
    LIBIBVERBS_LOADED=$(awk '$1 == "libibverbs.so.1" {print $3}' /tmp/efa-libfabric.ldd); \
    test "$(readlink -f "$LIBEFA_LOADED")" = \
        "$(readlink -f "${RUNTIME_LIB}/libefa.so.1.5.63.0")"; \
    test "$(readlink -f "$LIBIBVERBS_LOADED")" = \
        "$(readlink -f "${RUNTIME_LIB}/libibverbs.so.1.16.63.0")"; \
    ldd "$PROVIDER" | tee /tmp/efa-provider.ldd; \
    if grep -Fq "not found" /tmp/efa-provider.ldd; then exit 1; fi; \
    /opt/amazon/efa/bin/fi_info --version | tee /tmp/efa-fi-info.version; \
    grep -F "libfabric: 2.4.0amzn5.0" /tmp/efa-fi-info.version; \
    printf '%s\n' /opt/amazon/efa/lib > /etc/ld.so.conf.d/000_efa.conf; \
    ldconfig

ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

{% if not (framework == "sglang" and device == "cuda" and target == "runtime") %}
FROM aws_base AS aws_framework

USER root
{% endif %}

{% if framework == "trtllm" %}
# After the upstream mesonpy refactor, libplugin_LIBFABRIC.so lands under the
# Dynamo venv while the rest of the NIXL plugin set (GDS/UCX/POSIX) remains at
# the canonical arch-specific location. Copy LIBFABRIC alongside the others so
# NIXL_PLUGIN_DIR resolves every backend from a single directory, and expose a
# stable arch-agnostic alias at /opt/nvidia/nvda_nixl/plugins.
#
# Replace the upstream trtllm_runtime stage's bundled NIXL while preserving its
# libstdc++ preload. LIBFABRIC goes through libfabric directly (not UCX), so it
# is unaffected by the UCX 1.20.0 hang that the NIXL preload works around.
RUN --mount=from=wheel_builder,source=/opt/nvidia/nvda_nixl,target=/tmp/nvda_nixl \
    rm -rf /opt/nvidia/nvda_nixl && \
    cp -Pfr /tmp/nvda_nixl /opt/nvidia/nvda_nixl && \
    export LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/opt/nvidia/nvda_nixl/lib64/libnixl.so && \
    export NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins && \
    ldconfig

ENV LD_PRELOAD=/opt/dynamo/libstdc++.so.6:/opt/nvidia/nvda_nixl/lib64/libnixl.so
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

# Keep framework-specific overlays separate from the common publication gate.
FROM aws_base AS aws_framework

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
        if readelf --wide -S "$plugin" | grep -Fq .nv_fatbin; then exit 1; fi; \
        if readelf --version-info "$plugin" | grep -Fq FABRIC_1.9; then exit 1; fi; \
        for symbol in fi_getinfo fi_freeinfo fi_dupinfo; do \
            readelf --wide -Ws "$plugin" | \
                grep -E "[[:space:]]${symbol}@FABRIC_1[.]8([[:space:]]|$)"; \
        done; \
        env -u LD_PRELOAD \
            LD_LIBRARY_PATH="/tmp/cuda-stubs:/opt/amazon/efa/lib:${CORE}:${VENDOR}" \
            ldd -v "$plugin" | tee /tmp/nixl-plugin.ldd; \
        if grep -Fq "not found" /tmp/nixl-plugin.ldd; then exit 1; fi; \
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

# Keep aws as the public final-stage name used by the shared image workflow.
# For vLLM and TensorRT-LLM, fail the build unless the plugin that will be used
# at runtime loads eagerly and resolves libfabric from the pinned EFA prefix.
FROM aws_framework AS aws

USER root

{% if target == "runtime" and framework in ("vllm", "trtllm") %}
RUN set -eux; \
{% if framework == "vllm" %}
    PLUGIN=/opt/dynamo/nixl/plugins/libplugin_LIBFABRIC.so; \
    NIXL_LIB=/opt/dynamo/nixl; \
{% else %}
    PLUGIN=/opt/nvidia/nvda_nixl/lib64/plugins/libplugin_LIBFABRIC.so; \
    NIXL_LIB=/opt/nvidia/nvda_nixl/lib64; \
{% endif %}
    test -f "$PLUGIN"; \
    mkdir -p /tmp/cuda-stubs; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1; \
    env -u LD_PRELOAD \
        LD_LIBRARY_PATH="/tmp/cuda-stubs:/opt/amazon/efa/lib:${NIXL_LIB}:${LD_LIBRARY_PATH:-}" \
        ldd -v "$PLUGIN" 2>&1 | tee /tmp/nixl-libfabric.ldd; \
    if grep -Fq "not found" /tmp/nixl-libfabric.ldd; then exit 1; fi; \
    if grep -Fq "FABRIC_1.9" /tmp/nixl-libfabric.ldd; then exit 1; fi; \
    grep -F "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" \
        /tmp/nixl-libfabric.ldd; \
    env -u LD_PRELOAD \
        LD_LIBRARY_PATH="/tmp/cuda-stubs:/opt/amazon/efa/lib:${NIXL_LIB}:${LD_LIBRARY_PATH:-}" \
        PLUGIN="$PLUGIN" python3 -c \
            'import ctypes, os; ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)'
{% endif %}

LABEL com.nvidia.dynamo.efa-installer.version="{{ context.dynamo.efa_version }}" \
      com.nvidia.dynamo.efa-libfabric.version="2.4.0amzn5.0"

{% if target == "runtime" %}
USER dynamo
{% endif %}
