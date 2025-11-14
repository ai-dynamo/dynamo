# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

##################################
########## Build Arguments ########
##################################

# Base image configuration
ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
# TODO OPS-612: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
ARG BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

# Build configuration
ARG ENABLE_KVBM=false
ARG CARGO_BUILD_JOBS

# Define general architecture ARGs for supporting both x86 and aarch64 builds.
#   ARCH: Used for package suffixes (e.g., amd64, arm64)
#   ARCH_ALT: Used for Rust targets, manylinux suffix (e.g., x86_64, aarch64)
#
# Default values are for x86/amd64:
#   --build-arg ARCH=amd64 --build-arg ARCH_ALT=x86_64
#
# For arm64/aarch64, build with:
#   --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64
#TODO OPS-592: Leverage uname -m to determine ARCH instead of passing it as an arg
ARG ARCH=amd64
ARG ARCH_ALT=x86_64

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF=v1.19.0
ARG NIXL_REF=0.7.0
ARG NIXL_GDRCOPY_REF=v2.5.1

# Python configuration
ARG PYTHON_VERSION=3.12

##################################
########## Base Image ############
##################################

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Redeclare ARGs for this stage
ARG ARCH
ARG ARCH_ALT
ARG PYTHON_VERSION
ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
ARG NIXL_UCX_REF
ARG NIXL_REF
ARG NIXL_GDRCOPY_REF

USER root
WORKDIR /opt/dynamo

##################################
########## Tool Installation #####
##################################

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install SCCACHE if requested
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

# Set SCCACHE environment variables
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}} \
    RUSTC_WRAPPER=${USE_SCCACHE:+sccache} \
    CMAKE_C_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CXX_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CUDA_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache}

##################################
########## Rust Setup ############
##################################

# Rust environment setup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.90.0

# Define Rust target based on ARCH_ALT ARG
ARG RUSTARCH=${ARCH_ALT}-unknown-linux-gnu

# Install Rust
RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

##################################
########## External Services #####
##################################

# Install NATS server
ENV NATS_VERSION="v2.10.28"
RUN --mount=type=cache,target=/var/cache/apt \
    wget --tries=3 --waitretry=5 https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${ARCH}.deb && \
    dpkg -i nats-server-${NATS_VERSION}-${ARCH}.deb && rm nats-server-${NATS_VERSION}-${ARCH}.deb

# Install etcd
ENV ETCD_VERSION="v3.5.21"
RUN wget --tries=3 --waitretry=5 https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-${ARCH}.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=/usr/local/bin/etcd/:$PATH


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
ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
ARG NIXL_UCX_REF
ARG NIXL_REF
ARG NIXL_GDRCOPY_REF

WORKDIR /workspace

# Install system dependencies
RUN yum groupinstall -y 'Development Tools' &&  \
    dnf install -y almalinux-release-synergy && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y \
        # Build tools
        cmake \
        ninja-build \
        clang-devel \
        gcc-c++ \
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
        numactl-devel

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

# Set environment variables first so they can be used in COPY commands
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    PATH=/usr/local/cargo/bin:$PATH

# Copy artifacts from base stage
COPY --from=base $RUSTUP_HOME $RUSTUP_HOME
COPY --from=base $CARGO_HOME $CARGO_HOME

# Install SCCACHE if requested
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        /tmp/use-sccache.sh install; \
    fi

# Set SCCACHE environment variables
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}} \
    RUSTC_WRAPPER=${USE_SCCACHE:+sccache}

# Copy CUDA from base stage
COPY --from=base /usr/local/cuda /usr/local/cuda
COPY --from=base /etc/ld.so.conf.d/hpcx.conf /etc/ld.so.conf.d/hpcx.conf

ENV CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Create virtual environment for building wheels
ENV VIRTUAL_ENV=/workspace/.venv
RUN uv venv ${VIRTUAL_ENV} --python $PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf maturin[patchelf]

# Build and install gdrcopy
RUN git clone --depth 1 --branch ${NIXL_GDRCOPY_REF} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy/packages && \
    CUDA=/usr/local/cuda ./build-rpm-packages.sh && \
    rpm -Uvh gdrcopy-kmod-*.el8.noarch.rpm && \
    rpm -Uvh gdrcopy-*.el8.${ARCH_ALT}.rpm && \
    rpm -Uvh gdrcopy-devel-*.el8.noarch.rpm

# Build and install UCX
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    CC=${USE_SCCACHE:+sccache gcc} && \
    CXX=${USE_SCCACHE:+sccache g++} && \
    export CC=${CC} && \
    export CXX=${CXX} && \
    cd /usr/local/src && \
     git clone https://github.com/openucx/ucx.git && \
     cd ucx && 			     \
     git checkout $NIXL_UCX_REF &&	 \
     ./autogen.sh && ./configure     \
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

# build and install nixl
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone --depth 1 --branch ${NIXL_REF} "https://github.com/ai-dynamo/nixl.git" && \
    cd nixl && \
    mkdir build && \
    meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
    -Dcudapath_lib="/usr/local/cuda/lib64" \
    -Dcudapath_inc="/usr/local/cuda/include" \
    -Ducx_path="/usr/local/ucx" && \
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
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${ARCH}}" && \
    cd /workspace/nixl && \
    uv build . --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION

# Copy source code (order matters for layer caching)
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY launch/ /opt/dynamo/launch/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build dynamo wheels
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    cd /opt/dynamo && \
    uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    maturin build --release --out /opt/dynamo/dist && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        cd /opt/dynamo/lib/kvbm && \
        maturin build --release --out /opt/dynamo/dist; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo"

##############################################
########## Dev entrypoint image ##############
##############################################

FROM base AS dev

ARG ENABLE_KVBM
ARG ARCH_ALT

# Application environment variables
ENV DYNAMO_HOME=/opt/dynamo \
    CARGO_TARGET_DIR=/opt/dynamo/target

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # required for AIC perf files
        git \
        git-lfs \
        # rust build packages
        clang \
        libclang-dev \
        protobuf-compiler \
        # sudo for dev stage
        sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Add sudo privileges to dynamo user
    && echo "dynamo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dynamo \
    && chmod 0440 /etc/sudoers.d/dynamo

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache \
    && chown -R dynamo: /opt/dynamo /home/dynamo /workspace \
    && chmod -R g+w /opt/dynamo /home/dynamo/.cache /workspace

# Switch to dynamo user
USER dynamo
ENV HOME=/home/dynamo

# Create and activate virtual environment
ARG PYTHON_VERSION
RUN uv venv /opt/dynamo/venv --python $PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Install common and test dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    UV_GIT_LFS=1 uv pip install \
        --no-cache \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt

# NIXL environment variables
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

# Copy ucx and nixl libs
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo: --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo: --from=wheel_builder /opt/nvidia/nvda_nixl/lib64/. ${NIXL_LIB_DIR}/

# Copy built artifacts
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
COPY --chown=dynamo: --from=wheel_builder $CARGO_TARGET_DIR $CARGO_TARGET_DIR
COPY --chown=dynamo: --from=wheel_builder $CARGO_HOME $CARGO_HOME

COPY --chown=dynamo: ./ /workspace/

RUN uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        uv pip install /opt/dynamo/wheelhouse/kvbm*.whl; \
    fi \
    && cd /workspace/benchmarks \
    && UV_GIT_LFS=1 uv pip install --no-cache .

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Setup environment for all users
USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
