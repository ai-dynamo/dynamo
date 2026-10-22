{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/nixl_source.Dockerfile ===
##################################
##### nixl_builder ###############
##################################
# Builds UCX and NIXL (native + Python wheel) from source for XPU and CPU.
# CUDA images use the NIXL that ships in the upstream framework image or PyPI.

FROM wheel_builder_base AS nixl_builder

ARG TARGETARCH
ARG DEVICE
ARG USE_SCCACHE
ARG NIXL_UCX_REF
ARG NIXL_REF
ARG PYTHON_VERSION

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        git-lfs \
        libtool \
        meson \
        net-tools \
        ninja-build \
        pybind11-dev \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --reinstall --no-install-recommends \
        libibverbs-dev \
        rdma-core \
        ibverbs-utils \
        libibumad-dev \
        libnuma-dev \
        librdmacm-dev \
        ibverbs-providers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build and install UCX
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    : "${NIXL_UCX_REF:?}" && \
    git init -q ucx && cd ucx && \
    git remote add origin https://github.com/openucx/ucx.git && \
    git fetch --depth 1 origin "${NIXL_UCX_REF}" && \
    git checkout -q FETCH_HEAD && \
    # The intel/llm-scaler xe-GDR patch (ucx-v1.12.0.patch) is upstream since
    # UCX v1.21.0 (ib_md.c xe srcversion check, ze_copy_md.c HOST bit); restore
    # the fetch + git apply for DEVICE=xpu if this ref ever drops below v1.21.0.
    ./autogen.sh &&      \
    ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-verbs {{ "--with-ze --with-dm --with-efa " if device == "xpu" else "" }}\
        --without-cuda              \
        --enable-mt &&              \
    make -j &&                      \
    make -j install-strip &&        \
    /tmp/use-sccache.sh show-stats "UCX" && \
    echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
    echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
    ldconfig

{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
{% else %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
{% endif %}
{# XPU and CPU only support x86_64 #}
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/x86_64-linux-gnu
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins

# Build and install NIXL
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    source ${VIRTUAL_ENV}/bin/activate && \
    : "${NIXL_REF:?}" && \
    git init -q nixl && cd nixl && \
    git remote add origin https://github.com/ai-dynamo/nixl.git && \
    git fetch --depth 1 origin "${NIXL_REF}" && \
    git checkout -q FETCH_HEAD && \
    ./contrib/tomlutil.py --wheel-name "nixl-${DEVICE}" pyproject.toml && \
    mkdir build && \
    meson setup build/ --prefix="${NIXL_PREFIX}" --buildtype=release \
        -Ducx_path="/usr/local/ucx" && \
    cd build && \
    ninja && \
    ninja install && \
    /tmp/use-sccache.sh show-stats "NIXL"

ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

RUN echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Build NIXL wheel → /opt/dynamo/dist/nixl/nixl*.whl
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,id=uv-root-{{ context.dynamo.uv_version }},target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /workspace/nixl && \
    uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION
