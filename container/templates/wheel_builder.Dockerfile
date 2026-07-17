{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/wheel_builder.Dockerfile ===
##################################
##### Wheel Build Image ##########
##################################

{% if platform == "multi" and device == "cuda" %}
# Multi-arch: declare both manylinux base images with explicit --platform so each is
# always pulled as the correct native arch regardless of the current TARGETPLATFORM.
# BuildKit only fetches and builds the stage that TARGETARCH resolves to; the other
# is a no-op for each sub-build.
FROM --platform=linux/amd64 quay.io/pypa/manylinux_2_28_x86_64 AS manylinux_amd64
FROM --platform=linux/arm64 quay.io/pypa/manylinux_2_28_aarch64 AS manylinux_arm64
{% endif %}

##################################
##### wheel_builder_base #########
##################################
# Shared base for all wheel builds: tools, system deps, and native libraries (except nixl).

{% if platform == "multi" and device == "cuda" %}
FROM manylinux_${TARGETARCH} AS wheel_builder_base
{% else %}
FROM ${WHEEL_BUILDER_IMAGE} AS wheel_builder_base
{% endif %}

# Redeclare ARGs for this stage
ARG TARGETARCH
ARG CARGO_BUILD_JOBS
ARG DEVICE

WORKDIR /workspace

# Compliance: always create the rust license-harvest dir so the licenses stage's
# `COPY --from=wheel_builder /opt/dynamo/rust-licenses` never fails, even for
# targets that build no wheels. runtime_wheel_builder populates it post-build.
RUN mkdir -p /opt/dynamo/rust-licenses
{% if device == "xpu" or device == "cpu" %}
RUN apt clean && apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    curl ca-certificates zip unzip git lsb-release numactl wget vim \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
    libaio-dev \
    linux-libc-dev
{% endif %}

{% if device == "cuda" %}
# Copy CUDA from base stage
COPY --from=dynamo_base /usr/local/cuda /usr/local/cuda
COPY --from=dynamo_base /etc/ld.so.conf.d/hpcx.conf /etc/ld.so.conf.d/hpcx.conf
{% endif %}

# Set environment variables first so they can be used in COPY commands
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    PATH=/usr/local/cargo/bin:$PATH



# Copy artifacts from base stage
COPY --from=dynamo_base $RUSTUP_HOME $RUSTUP_HOME
COPY --from=dynamo_base $CARGO_HOME $CARGO_HOME

{% if device == "xpu" %}
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

# Fetch UCX patch
RUN wget --tries=3 --waitretry=5 https://raw.githubusercontent.com/intel/llm-scaler/35a14cbc08d714f460a29b7a7328df5620c8530f/vllm/patches/ai-dynamo-xpu/patches/ucx-v1.12.0.patch -O /tmp/ucx.patch

# Install Intel GPU runtime packages
RUN apt update -y && \
    apt-get install -y intel-opencl-icd  \
    libze-intel-gpu-raytracing intel-ocloc intel-oneapi-compiler-dpcpp-cpp-2025.3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
{% endif %}

{% if device == "xpu" or device == "cpu" %}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # NIXL build dependencies
        autoconf \
        automake \
        cmake \
        git-lfs \
        libtool \
        meson \
        net-tools \
        ninja-build \
        pybind11-dev \
        # Rust build dependencies
        clang \
        libclang-dev \
        protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y \
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
{% endif %}

{% if device == "cuda" %}
# Install system dependencies
# Cache dnf downloads; sharing=locked avoids dnf/rpm races with concurrent builds.
# --setopt=tsflags=nocontexts: skip SELinux file-context labeling. The manylinux
# image lacks the SELinux policy store that some compute nodes expect; without
# this flag, dnf fails with "ValueError: SELinux policy is not managed".
RUN --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    dnf install -y --setopt=tsflags=nocontexts almalinux-release-synergy && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y --setopt=tsflags=nocontexts \
        # Autotools (required for UCX, libfabric ./autogen.sh and ./configure)
        autoconf \
        automake \
        libtool \
        make \
        # RPM build tools (required for gdrcopy's build-rpm-packages.sh)
        rpm-build \
        rpm-sign \
        # Build tools
        cmake \
        ninja-build \
        clang-devel \
        # Install GCC toolset 14 (CUDA compatible, max version 14)
        gcc-toolset-14-gcc \
        gcc-toolset-14-gcc-c++ \
        gcc-toolset-14-binutils \
        flex \
        wget \
        # Kernel module build dependencies
        dkms \
        # Protobuf support
        protobuf-compiler \
        # RDMA/InfiniBand support (required for UCX build with --with-verbs)
        libibverbs \
        libibverbs-devel \
        rdma-core \
        rdma-core-devel \
        libibumad \
        libibumad-devel \
        librdmacm-devel \
        numactl-devel \
        # Libfabric support
        libcurl-devel \
        openssl-devel \
        libuuid-devel \
        zlib-devel

# Default comes from context.yaml; keep it in sync with upstream NIXL's
# contrib/Dockerfile.manylinux. NIXL v1.0.x needs newer hwloc than RHEL8 ships.
ARG HWLOC_VERSION
RUN cd /tmp && \
    HWLOC_SERIES="$(echo "${HWLOC_VERSION}" | cut -d. -f1,2)" && \
    wget -q "https://download.open-mpi.org/release/hwloc/v${HWLOC_SERIES}/hwloc-${HWLOC_VERSION}.tar.gz" && \
    tar -xzf "hwloc-${HWLOC_VERSION}.tar.gz" && \
    cd "hwloc-${HWLOC_VERSION}" && \
    ./configure --prefix=/usr/local --disable-nvml && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf "/tmp/hwloc-${HWLOC_VERSION}" "/tmp/hwloc-${HWLOC_VERSION}.tar.gz"

# Set GCC toolset 14 as the default compiler (CUDA requires GCC <= 14)
ENV PATH="/opt/rh/gcc-toolset-14/root/usr/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib64:${LD_LIBRARY_PATH}" \
    CC="/opt/rh/gcc-toolset-14/root/usr/bin/gcc" \
    CXX="/opt/rh/gcc-toolset-14/root/usr/bin/g++"
{% endif %}

# Ensure a modern protoc is available (required for --experimental_allow_proto3_optional)
RUN set -eux; \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64"); \
    PROTOC_VERSION=25.3; \
    case "${ARCH_ALT}" in \
      x86_64) PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-x86_64.zip" ;; \
      aarch64) PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-aarch_64.zip" ;; \
      *) echo "Unsupported architecture: ${ARCH_ALT}" >&2; exit 1 ;; \
    esac; \
    wget --tries=3 --waitretry=5 -O /tmp/protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}"; \
    rm -f /usr/local/bin/protoc /usr/bin/protoc; \
    unzip -o /tmp/protoc.zip -d /usr/local bin/protoc include/*; \
    chmod +x /usr/local/bin/protoc; \
    ln -s /usr/local/bin/protoc /usr/bin/protoc; \
    protoc --version

# Point build tools explicitly at the modern protoc
ENV PROTOC=/usr/local/bin/protoc

{% if device == "xpu" or device == "cpu" %}
# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}
{% else %}
ENV CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-} \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% endif %}

# Create virtual environment for building wheels
ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/workspace/.venv
# Cache uv downloads; uv handles its own locking for this cache.
# pyyaml: needed by the compliance NOTICES-bundling steps below (overrides.py
# imports yaml at module scope); the system python3 doesn't ship it.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv venv ${VIRTUAL_ENV} --python $PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf maturin[patchelf] tomlkit pyyaml

ARG NIXL_UCX_REF

{% if device == "cuda" %}
ARG NIXL_GDRCOPY_REF

# Build and install gdrcopy
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    git clone --depth 1 --branch ${NIXL_GDRCOPY_REF} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy/packages && \
    CUDA=/usr/local/cuda ./build-rpm-packages.sh && \
    rpm -Uvh gdrcopy-kmod-*.el8.noarch.rpm && \
    rpm -Uvh gdrcopy-*.el8.${ARCH_ALT}.rpm && \
    rpm -Uvh gdrcopy-devel-*.el8.noarch.rpm
{% endif %}

# sccache binary is pre-installed in dynamo_base; stage it off-PATH so
# Meson doesn't auto-detect it as a CUDA compiler launcher
# (https://github.com/mesonbuild/meson/issues/11118).
# When USE_SCCACHE=true the RUN below symlinks it onto PATH before install.
COPY --from=dynamo_base /usr/local/bin/sccache /opt/sccache/sccache

ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        ln -s /opt/sccache/sccache /usr/local/bin/sccache && \
        /tmp/use-sccache.sh install; \
    fi

# Compliance: native source archives drop here. RUN git clone / wget …tar lines
# in the wheel_builder pipeline preserve their resulting archive at
# /tmp/native-sources/<name>-<version>.tar.gz so the per-image `sources_collect`
# stage can COPY them out for OSRB submission. Created here unconditionally
# (cheap) so the COPY always succeeds even when no native source builds run
# for this framework.
RUN mkdir -p /tmp/native-sources

# Compliance source-archival pattern (do NOT add ARG ENABLE_SOURCE_ARCHIVAL
# at this scope — it would invalidate every downstream layer when the flag
# flips between PR builds and post-merge builds).
#
# When future work adds cargo-vendor / go-mod-vendor / native source-tree
# preservation, declare the ARG INLINE in the smallest possible scope,
# immediately before the gated RUN, e.g.:
#
#     ARG ENABLE_SOURCE_ARCHIVAL=false
#     RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
#           cargo vendor --locked --manifest-path /opt/dynamo/Cargo.toml \
#               /tmp/native-sources/rust-vendor; \
#         fi
#
# This way the cache invalidation is contained to one RUN layer (the gated
# one), not the rest of wheel_builder_base. shared-build-image.yml passes
# ENABLE_SOURCE_ARCHIVAL=true via extra_build_args on push / release /
# workflow_dispatch events; PR builds get the default "false" and skip.

# Set SCCACHE environment variables (RUSTC_WRAPPER is set dynamically by
# setup-env only when the sccache server starts successfully)
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}

# Always build FFmpeg so libs are available for Rust checks in CI.
# vLLM uses a narrow VP8/VP9/raw-video decode surface and VP9 encoding. Other
# frameworks retain their existing media configuration.
# Do not treat a hardware implementation as changing a codec's patent status:
# the vLLM build deliberately does not compile the NVENC H.264 encoder.
# Do not delete the source tarball for legal reasons.
ARG FFMPEG_VERSION
ARG NV_CODEC_HEADERS_REF
ARG LIBVPX_REF
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    if [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then \
    apt-get update -y && apt-get install -y build-essential pkg-config xz-utils git yasm; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    elif [ "$DEVICE" = "cuda" ]; then \
    dnf install -y --setopt=tsflags=nocontexts pkg-config xz git yasm; \
    fi && \
    cd /tmp && \
{% if framework != "vllm" %}
    # nv-codec-headers: provides the NVENC/NVDEC API headers ffmpeg compiles against.
    # Header-only, no runtime dep here; libcuda/libnvidia-encode are loaded at runtime
    # in the consuming container.
    git clone --depth 1 --branch ${NV_CODEC_HEADERS_REF} https://github.com/FFmpeg/nv-codec-headers.git && \
    make -C nv-codec-headers PREFIX=/usr/local install && \
{% endif %}
    # libvpx: BSD-licensed VP9 encoder needed for the WebM output path. Built from
    # source so we don't need to track distro package names (libvpx-dev on Debian
    # vs libvpx-devel via EPEL on RHEL/manylinux).
    git clone --depth 1 --branch ${LIBVPX_REF} https://chromium.googlesource.com/webm/libvpx.git && \
    cd libvpx && \
    ./configure --prefix=/usr/local --enable-shared --disable-static --disable-examples --disable-unit-tests --disable-tools --disable-docs && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && \
    curl --retry 5 --retry-delay 3 -LO https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    tar xf ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    cd ffmpeg-${FFMPEG_VERSION} && \
    ./configure \
        --prefix=/usr/local \
        --disable-gpl \
        --disable-nonfree \
        --disable-doc \
        --disable-static \
        --disable-x86asm \
        --disable-network \
        --disable-bsfs \
        --disable-devices \
        --disable-libdrm \
        --enable-shared \
{% if framework == "vllm" %}
        --disable-decoders \
        --enable-decoder=vp8,vp9,rawvideo \
        --disable-demuxers \
        --enable-demuxer=mov,matroska,rawvideo \
        --disable-parsers \
        --enable-parser=vp8,vp9 \
        --enable-libvpx \
        --disable-encoders \
        --enable-encoder=libvpx_vp9 \
        --disable-muxers \
        --enable-muxer=matroska,webm \
        --disable-protocols \
        --enable-protocol=file,pipe && \
{% else %}
        --enable-nvenc \
        --enable-libvpx \
        --disable-encoders \
        --enable-encoder=h264_nvenc,libvpx_vp9 \
        --disable-muxers \
        --enable-muxer=mov,mp4,matroska,webm \
        --enable-protocol=file,pipe && \
{% endif %}
    make -j$(nproc) && \
    make install && \
    /tmp/use-sccache.sh show-stats "FFMPEG" && \
    ldconfig && \
    mkdir -p /usr/local/src/ffmpeg && \
    find /tmp/ffmpeg-${FFMPEG_VERSION} \( -name config.log -o -name config.status \) -delete && \
    mv /tmp/ffmpeg-${FFMPEG_VERSION}* /usr/local/src/ffmpeg/

# Build and install UCX
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    git clone https://github.com/openucx/ucx.git && \
    cd ucx &&  \
    git checkout $NIXL_UCX_REF &&	 \
    if [ "$DEVICE" = "xpu" ]; then \
    git apply --ignore-whitespace /tmp/ucx.patch; \
    fi && \
    ./autogen.sh &&      \
    if [ "$DEVICE" = "xpu" ]; then \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --with-ze                   \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-verbs                \
        --with-dm                   \
        --with-efa                  \
        --without-cuda              \
        --enable-mt;                 \
    elif [ "$DEVICE" = "cuda" ]; then \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-cuda=/usr/local/cuda \
        --with-verbs                \
        --with-dm                   \
        --with-gdrcopy=/usr/local   \
        --with-efa                  \
        --enable-mt;                 \
    elif [ "$DEVICE" = "cpu" ]; then  \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-verbs                \
        --without-cuda              \
        --enable-mt;                 \
     fi && \
     make -j &&                      \
     make -j install-strip &&        \
     /tmp/use-sccache.sh show-stats "UCX" && \
     echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
     echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
     ldconfig

{% if device == "cuda" %}
ARG NIXL_LIBFABRIC_REPO
ARG NIXL_LIBFABRIC_REF
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    git clone "${NIXL_LIBFABRIC_REPO}" && \
    cd libfabric && \
    git checkout $NIXL_LIBFABRIC_REF && \
    ./autogen.sh && \
    ./configure --prefix="/usr/local/libfabric" \
                --disable-verbs \
                --disable-psm3 \
                --disable-opx \
                --disable-usnic \
                --disable-rstream \
                --enable-efa \
                --with-cuda=/usr/local/cuda \
                --enable-cuda-dlopen \
                --with-gdrcopy \
                --enable-gdrcopy-dlopen && \
    make -j$(nproc) && \
    make install && \
    /tmp/use-sccache.sh show-stats "LIBFABRIC" && \
    echo "/usr/local/libfabric/lib" > /etc/ld.so.conf.d/libfabric.conf && \
    ldconfig

ENV PKG_CONFIG_PATH="/usr/local/libfabric/lib/pkgconfig:${PKG_CONFIG_PATH}"
{% endif %}

{% if framework == "vllm" and device == "cuda" %}
# Build and install AWS SDK C++ (required for NIXL OBJ backend / S3 support)
ARG AWS_SDK_CPP_VERSION
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    git clone --recurse-submodules --depth 1 --branch ${AWS_SDK_CPP_VERSION} \
        https://github.com/aws/aws-sdk-cpp.git /tmp/aws-sdk-cpp && \
    mkdir -p /tmp/aws-sdk-cpp/build && \
    cd /tmp/aws-sdk-cpp/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_ONLY="s3" \
        -DENABLE_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/aws-sdk-cpp && \
    ldconfig && \
    /tmp/use-sccache.sh show-stats "AWS SDK C++"
{% endif %}


##################################
##### runtime_wheel_builder ######
##################################
# Builds ai-dynamo, ai-dynamo-runtime, and gpu_memory_service wheels, sans nixl.

FROM wheel_builder_base AS runtime_wheel_builder

{% if target not in ("dev", "local-dev") %}
# Copy source code (order matters for layer caching)
COPY .cargo/ /opt/dynamo/.cargo/
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build ai-dynamo (pure Python) and ai-dynamo-runtime (maturin) wheels
ARG USE_SCCACHE
ARG ENABLE_MEDIA_FFMPEG
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    mkdir -p ${CARGO_TARGET_DIR} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    cd /opt/dynamo && \
    uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then \
        maturin build --release --features "media-ffmpeg,kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass" --out /opt/dynamo/dist; \
    else \
        maturin build --release --features "kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass" --out /opt/dynamo/dist; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo Runtime"

# Compliance: harvest each crate's real LICENSE files from the cargo registry
# source cache so the rust NOTICES generator can inline upstream license text
# (the runtime image keeps only the compiled wheel). Keyed "<name>-<version>"
# to match generators/rust.py. Best-effort: unreadable/absent files are skipped
# and the generator falls back to canonical SPDX text. cargo's registry lives
# under CARGO_HOME and/or the cache-mounted /root/.cargo — scan both.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    for src in "${CARGO_HOME}/registry/src" /root/.cargo/registry/src; do \
        [ -d "$src" ] || continue; \
        find "$src" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r crate; do \
            dest="/opt/dynamo/rust-licenses/$(basename "$crate")"; \
            for lf in "$crate"/LICENSE* "$crate"/LICENCE* "$crate"/COPYING* "$crate"/NOTICE* "$crate"/UNLICENSE*; do \
                [ -e "$lf" ] || continue; \
                mkdir -p "$dest" && cp "$lf" "$dest/" 2>/dev/null || true; \
            done; \
        done; \
    done; \
    echo "rust license harvest: $(find /opt/dynamo/rust-licenses -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) crates with license files"; \
    true

# Compliance: bundle the human-readable third-party Rust NOTICES into the
# maturin wheels themselves (PEP 639 <dist-info>/licenses/), using the harvested
# crate license texts. The wheel already carries maturin's CycloneDX SBOM (the
# machine-readable inventory); this adds the texts the redistributed wheel's
# MIT/BSD/Apache attribution clauses require. Best-effort + non-fatal: a failure
# leaves the wheel with its SBOM intact rather than breaking the build.
# Must run with the build venv's python: bundle_wheel_notices imports
# compliance.overrides, which needs pyyaml (installed in the venv above);
# the bare system python3 lacks it and the step would no-op with a warning.
COPY container/compliance /opt/compliance
RUN set -u; injected=0; \
    for whl in /opt/dynamo/dist/ai_dynamo_runtime*.whl; do \
        [ -e "$whl" ] || continue; \
        PYTHONPATH=/opt ${VIRTUAL_ENV}/bin/python3 -m compliance.bundle_wheel_notices \
            --wheel "$whl" --licenses-dir /opt/dynamo/rust-licenses -v \
            && injected=$((injected+1)) || echo "::warning::wheel NOTICES bundling failed for $whl (SBOM retained)"; \
    done; \
    echo "wheel NOTICES bundled into $injected wheel(s)"

# Compliance source archival: vendor the workspace lockfile for the OSRB
# bundle. Gated on ENABLE_SOURCE_ARCHIVAL so PR builds skip the ~200-400 MB
# vendor pull. The vendor tree is consumed downstream by each runtime
# template's sources_collect stage, which filters against the installed
# wheels' embedded SBOMs to keep only the third-party crates we actually
# ship. Stay scoped to one RUN layer (cache invalidation contained).
ARG ENABLE_SOURCE_ARCHIVAL=false
# Mount cargo registry + git caches so re-runs don't re-download the
# ~750 crates from crates.io every build. `sharing=shared` lets parallel
# builds (e.g. multiple frameworks in CI) read the same cache concurrently.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        mkdir -p /tmp/dynamo-vendor-full && \
        cd /opt/dynamo && \
        cargo vendor --locked /tmp/dynamo-vendor-full > /dev/null && \
        cp Cargo.toml Cargo.lock /tmp/dynamo-vendor-full/ ; \
    fi

{% else %}
# Dev/local-dev targets do not have pre-built wheels or /workspace source code.
# After you start the local-dev/dev container, you will need to build from source:
#   cargo build --features dynamo-llm/block-manager
#   cd /workspace/lib/bindings/python && maturin develop --uv && cd /workspace
#   uv pip install --no-deps -e /workspace
# See container/launch_message/dev.txt for the full setup steps.

# Create dist dir with a placeholder so downstream COPY --from=wheel_builder /opt/dynamo/dist/*.whl always has a match.
RUN mkdir -p /opt/dynamo/dist ${CARGO_TARGET_DIR} && \
    touch /opt/dynamo/dist/.placeholder.whl

# Dev/local-dev skip the full COPY lib/ above, so copy gpu_memory_service source explicitly for the wheel build below
COPY lib/gpu_memory_service/ /opt/dynamo/lib/gpu_memory_service/
{% endif %}

# Build gpu-memory-service wheel → /opt/dynamo/dist/gpu_memory_service*.whl (small C++ extension, fast build -- all targets, all frameworks)
{% if device == "cuda" %}
# Build gpu_memory_service wheel (C++ extension only needs Python headers, no CUDA/torch)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    if [ "$ENABLE_GPU_MEMORY_SERVICE" = "true" ]; then \
        export UV_CACHE_DIR=/root/.cache/uv && \
        source ${VIRTUAL_ENV}/bin/activate && \
        uv build --wheel --out-dir /opt/dynamo/dist /opt/dynamo/lib/gpu_memory_service; \
    fi
{% endif %}


##################################
##### wheel_builder ##############
##################################
{% if "nixl_ref" in context[framework] or device == "xpu" %}
# Builds NIXL (native + Python wheel) and NIXL-linked extension wheels, then
# consolidates all wheels.
# Runtime templates COPY from this stage.
# Note: XPU triggers this path even when the framework section lacks nixl_ref,
# because no upstream XPU runtime image ships pre-built NIXL.

FROM wheel_builder_base AS wheel_builder

# Build and install nixl
ARG TARGETARCH
ARG DEVICE
ARG NIXL_REF
ARG USE_SCCACHE
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}

RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone "https://github.com/ai-dynamo/nixl.git" && \
    cd nixl && \
    git checkout ${NIXL_REF} && \
    if [ "$DEVICE" = "cuda" ]; then \
        PKG_NAME="nixl-cu${CUDA_MAJOR}"; \
    else \
        PKG_NAME="nixl-${DEVICE}"; \
    fi && \
    ./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml && \
    mkdir build && \
    if [ "$DEVICE" = "cuda" ]; then \
        meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
            -Dcudapath_lib="/usr/local/cuda/lib64" \
            -Dcudapath_inc="/usr/local/cuda/include" \
            -Ducx_path="/usr/local/ucx" \
            -Dlibfabric_path="/usr/local/libfabric"; \
    elif [ "$DEVICE" = "xpu" ]; then \
        meson setup build/ --prefix=/opt/intel/intel_nixl --buildtype=release \
            -Ducx_path="/usr/local/ucx"; \
    elif [ "$DEVICE" = "cpu" ]; then \
        meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
            -Ducx_path="/usr/local/ucx"; \
    fi && \
    cd build && \
    ninja && \
    ninja install && \
    /tmp/use-sccache.sh show-stats "NIXL"

{% if device == "xpu" %}
{# XPU only supports x86_64; no ARCH_ALT ARG needed #}
ENV NIXL_LIB_DIR=/opt/intel/intel_nixl/lib/x86_64-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/intel/intel_nixl/lib/x86_64-linux-gnu/plugins \
    NIXL_PREFIX=/opt/intel/intel_nixl
{% elif device == "cpu" %}
ENV NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl
{% else %}
ENV NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl
{% endif %}

ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

RUN echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

{% if not (framework == "sglang" and device == "cuda" and target in ("runtime", "dev", "local-dev")) %}
# Build NIXL wheel → /opt/dynamo/dist/nixl/nixl*.whl (C++ transport library, all targets)
ARG PYTHON_VERSION
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /workspace/nixl && \
    uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Copy source code (order matters for layer caching)
COPY .cargo/ /opt/dynamo/.cargo/
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build kvbm wheel (with nixl linkage via auditwheel repair)
ARG ENABLE_KVBM
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    --mount=type=cache,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    mkdir -p ${CARGO_TARGET_DIR} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        cd /opt/dynamo/lib/bindings/kvbm && \
        KVBM_FEATURES=""; \
        if [ "$DEVICE" = "cuda" ]; then KVBM_FEATURES="--features nccl"; fi && \
        maturin build --release ${KVBM_FEATURES} --out target/wheels && \
        if [ "$DEVICE" = "cuda" ]; then \
            auditwheel repair \
                --exclude libnixl.so \
                --exclude libnixl_build.so \
                --exclude libnixl_common.so \
                --exclude 'lib*.so*' \
                --plat manylinux_2_28_${ARCH_ALT} \
                --wheel-dir /opt/dynamo/dist \
                target/wheels/*.whl; \
        elif [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then \
            cp target/wheels/*.whl /opt/dynamo/dist/; \
        fi; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo KVBM"
{% endif %}

# Consolidate all wheels from the runtime wheel builder stage
COPY --from=runtime_wheel_builder /opt/dynamo/dist/ /opt/dynamo/dist/

# Compliance: bundle third-party Rust NOTICES into the kvbm wheel built in this
# stage (the ai-dynamo-runtime wheel was already bundled in runtime_wheel_builder
# and arrives consolidated above). Harvest kvbm's crate licenses from the cargo
# registry, then inject into its auditwheel-repaired wheel. Best-effort/non-fatal.
# Venv python required: compliance.overrides needs pyyaml, absent from the
# system python3 (see the runtime_wheel_builder bundling step).
COPY container/compliance /opt/compliance
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    set -u; \
    for src in "${CARGO_HOME}/registry/src" /root/.cargo/registry/src; do \
        [ -d "$src" ] || continue; \
        find "$src" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r crate; do \
            dest="/opt/dynamo/rust-licenses/$(basename "$crate")"; \
            for lf in "$crate"/LICENSE* "$crate"/LICENCE* "$crate"/COPYING* "$crate"/NOTICE* "$crate"/UNLICENSE*; do \
                [ -e "$lf" ] || continue; mkdir -p "$dest" && cp "$lf" "$dest/" 2>/dev/null || true; \
            done; \
        done; \
    done; \
    for whl in /opt/dynamo/dist/kvbm*.whl; do \
        [ -e "$whl" ] || continue; \
        PYTHONPATH=/opt ${VIRTUAL_ENV}/bin/python3 -m compliance.bundle_wheel_notices \
            --wheel "$whl" --licenses-dir /opt/dynamo/rust-licenses -v \
            || echo "::warning::kvbm wheel NOTICES bundling failed (SBOM retained)"; \
    done; \
    echo "kvbm wheel NOTICES step done"

{% else %}
# SGLang CUDA uses NIXL from the upstream lmsysorg/sglang runtime image and
# does not build Dynamo KVBM. Keep this alias so downstream stages can still
# COPY Dynamo wheels and build tools from a common wheel_builder stage name.
# SGLang dev/source builds may link nixl-sys against stubs when native NIXL is
# absent; block-manager/KVBM runtime work should use vllm/trtllm/none images.
FROM runtime_wheel_builder AS wheel_builder
{% endif %}
