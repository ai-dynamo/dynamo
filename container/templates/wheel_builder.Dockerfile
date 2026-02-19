{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/wheel_builder.Dockerfile ===
##################################
##### Wheel Build Image ##########
##################################

# Redeclare ARCH_ALT ARG so it's available for interpolation in the FROM instruction
ARG ARCH_ALT

FROM quay.io/pypa/manylinux_2_28_${ARCH_ALT} AS wheel_builder

# Redeclare ARGs for this stage
ARG ARCH
ARG ARCH_ALT
ARG CARGO_BUILD_JOBS

WORKDIR /workspace

# Copy CUDA from base stage
COPY --from=dynamo_base /usr/local/cuda /usr/local/cuda
COPY --from=dynamo_base /etc/ld.so.conf.d/hpcx.conf /etc/ld.so.conf.d/hpcx.conf

# Set environment variables first so they can be used in COPY commands
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    PATH=/usr/local/cargo/bin:$PATH

# Copy artifacts from base stage
COPY --from=dynamo_base $RUSTUP_HOME $RUSTUP_HOME
COPY --from=dynamo_base $CARGO_HOME $CARGO_HOME

# Install system dependencies
# Cache dnf downloads; sharing=locked avoids dnf/rpm races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    dnf install -y almalinux-release-synergy && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y \
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
        hwloc \
        hwloc-devel \
        libcurl-devel \
        openssl-devel \
        libuuid-devel \
        zlib-devel

# Set GCC toolset 14 as the default compiler (CUDA requires GCC <= 14)
ENV PATH="/opt/rh/gcc-toolset-14/root/usr/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib64:${LD_LIBRARY_PATH}" \
    CC="/opt/rh/gcc-toolset-14/root/usr/bin/gcc" \
    CXX="/opt/rh/gcc-toolset-14/root/usr/bin/g++"


# Ensure a modern protoc is available (required for --experimental_allow_proto3_optional)
RUN set -eux; \
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

ENV CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Create virtual environment for building wheels
ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/workspace/.venv
# Cache uv downloads; uv handles its own locking for this cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv venv ${VIRTUAL_ENV} --python $PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf maturin[patchelf] tomlkit

ARG NIXL_UCX_REF
ARG NIXL_REF
ARG NIXL_GDRCOPY_REF

# Build and install gdrcopy
RUN git clone --depth 1 --branch ${NIXL_GDRCOPY_REF} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy/packages && \
    CUDA=/usr/local/cuda ./build-rpm-packages.sh && \
    rpm -Uvh gdrcopy-kmod-*.el8.noarch.rpm && \
    rpm -Uvh gdrcopy-*.el8.${ARCH_ALT}.rpm && \
    rpm -Uvh gdrcopy-devel-*.el8.noarch.rpm

# Install SCCACHE if requested
ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

# Set SCCACHE environment variables (RUSTC_WRAPPER is set dynamically by
# setup-env only when the sccache server starts successfully)
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}

# Build FFmpeg from source
# Do not delete the source tarball for legal reasons
ARG FFMPEG_VERSION
ARG ENABLE_MEDIA_FFMPEG
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    dnf install -y pkg-config && \
    cd /tmp && \
    curl -LO https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    tar xf ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    cd ffmpeg-${FFMPEG_VERSION} && \
    ./configure \
        --prefix=/usr/local \
        --disable-gpl \
        --disable-nonfree \
        --disable-programs \
        --disable-doc \
        --disable-static \
        --disable-x86asm \
        --disable-postproc \
        --disable-network \
        --disable-encoders \
        --disable-muxers \
        --disable-bsfs \
        --disable-devices \
        --disable-libdrm \
        --enable-shared && \
    make -j$(nproc) && \
    make install && \
    /tmp/use-sccache.sh show-stats "FFMPEG" && \
    ldconfig && \
    mkdir -p /usr/local/src/ffmpeg && \
    mv /tmp/ffmpeg-${FFMPEG_VERSION}* /usr/local/src/ffmpeg/; \
fi

# Build and install UCX
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
     git clone https://github.com/openucx/ucx.git && \
     cd ucx && 			     \
     git checkout $NIXL_UCX_REF &&	 \
     ./autogen.sh &&      \
     ./contrib/configure-release    \
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
        --enable-mt &&              \
     make -j &&                      \
     make -j install-strip &&        \
     /tmp/use-sccache.sh show-stats "UCX" && \
     echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
     echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
     ldconfig

ARG NIXL_LIBFABRIC_REF
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    git clone https://github.com/ofiwg/libfabric.git && \
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

{% if framework == "vllm" %}
# Build and install AWS SDK C++ (required for NIXL OBJ backend / S3 support)
ARG AWS_SDK_CPP_VERSION=1.11.581
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
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

# build and install nixl
ARG CUDA_MAJOR
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone "https://github.com/ai-dynamo/nixl.git" && \
    cd nixl && \
    git checkout ${NIXL_REF} && \
    PKG_NAME="nixl-cu${CUDA_MAJOR}" && \
    ./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml && \
    mkdir build && \
    meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
    -Dcudapath_lib="/usr/local/cuda/lib64" \
    -Dcudapath_inc="/usr/local/cuda/include" \
    -Ducx_path="/usr/local/ucx" \
    -Dlibfabric_path="/usr/local/libfabric" && \
    cd build && \
    ninja && \
    ninja install && \
    /tmp/use-sccache.sh show-stats "NIXL"

ENV NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64  \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

RUN echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /workspace/nixl && \
    uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION

# Copy source code (order matters for layer caching)
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build dynamo wheels. The caches do not need the "shared" lock because Cargo has its own locking mechanism.
ARG ENABLE_KVBM
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    mkdir -p ${CARGO_TARGET_DIR} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    cd /opt/dynamo && \
    uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then \
        maturin build --release --features "media-ffmpeg" --out /opt/dynamo/dist; \
    else \
        maturin build --release --out /opt/dynamo/dist; \
    fi && \
    if [ "$ENABLE_KVBM" == "true" ]; then \
        cd /opt/dynamo/lib/bindings/kvbm && \
        maturin build --release --out target/wheels && \
        auditwheel repair \
            --exclude libnixl.so \
            --exclude libnixl_build.so \
            --exclude libnixl_common.so \
            --exclude 'lib*.so*' \
            --plat manylinux_2_28_${ARCH_ALT} \
            --wheel-dir /opt/dynamo/dist \
            target/wheels/*.whl; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo"


# Build gpu_memory_service wheel (C++ extension only needs Python headers, no CUDA/torch)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$ENABLE_GPU_MEMORY_SERVICE" = "true" ]; then \
        export UV_CACHE_DIR=/root/.cache/uv && \
        source ${VIRTUAL_ENV}/bin/activate && \
        uv build --wheel --out-dir /opt/dynamo/dist /opt/dynamo/lib/gpu_memory_service; \
    fi
