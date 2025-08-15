# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
# TODO OPS-612: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
ARG BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"
ARG RELEASE_BUILD=false
ARG ENABLE_KVBM=false

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


##################################
########## Base Image ############
##################################

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Redeclare ARCH and ARCH_ALT so they're available in this stage
ARG ARCH
ARG ARCH_ALT
ARG CARGO_BUILD_JOBS

ARG NIXL_UCX_REF=v1.19.x
ARG NIXL_REF=0.4.1

# Environment variables for NIXL
ENV NIXL_SRC_DIR=/opt/nixl \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib/${ARCH_ALT}-linux-gnu/plugins

USER root
ARG PYTHON_VERSION=3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Rust environment setup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.87.0

WORKDIR /opt/dynamo

# Define Rust target based on ARCH_ALT ARG
ARG RUSTARCH=${ARCH_ALT}-unknown-linux-gnu

# Install Rust using RUSTARCH derived from ARCH_ALT
RUN wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    # TODO OPS-591: Add SHA check back based on RUSTARCH
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # NIXL build dependencies
        autoconf \
        automake \
        cmake \
        git \
        libtool \
        meson \
        net-tools \
        ninja-build \
        pybind11-dev \
        # These headers are missing with the hpcx installer, required
        # by UCX to find RDMA devices
        ibverbs-providers \
        ibverbs-utils \
        libibumad-dev \
        libibverbs-dev \
        librdmacm-dev \
        libnuma-dev \
        rdma-core \
        # Rust build dependencies
        clang \
        libclang-dev \
        protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download external dependencies in parallel for better performance
ENV NATS_VERSION="v2.10.28"
RUN --mount=type=cache,target=/var/cache/apt \
    wget --tries=3 --waitretry=5 https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${ARCH}.deb && \
    dpkg -i nats-server-${NATS_VERSION}-${ARCH}.deb && rm nats-server-${NATS_VERSION}-${ARCH}.deb

ENV ETCD_VERSION="v3.5.21"
RUN wget --tries=3 --waitretry=5 https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-${ARCH}.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=/usr/local/bin/etcd/:$PATH

### UCX EFA Setup ###
RUN rm -rf /opt/hpcx/ucx && \
    rm -rf /usr/local/ucx && \
    echo "Building UCX with reference $NIXL_UCX_REF" && \
    cd /usr/local/src && \
    git clone --depth 1 --branch $NIXL_UCX_REF https://github.com/openucx/ucx.git && \
    cd ucx && \
    ./autogen.sh && \
    ./configure \
      --prefix=/usr/local/ucx \
      --enable-shared \
      --disable-static \
      --disable-doxygen-doc \
      --enable-optimizations \
      --enable-cma \
      --enable-devel-headers \
      --with-cuda=/usr/local/cuda \
      --with-verbs \
      --with-efa \
      --with-dm \
      --with-gdrcopy=/usr/local \
      --enable-mt && \
    make -j$(nproc) &&                              \
    make -j$(nproc) install-strip &&                \
    echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
    echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
    ldconfig && \
    cd /usr/local/src && \
    rm -rf ucx

# UCX environment variables
ENV CPATH=/usr/include:$CPATH \
    PATH=/usr/bin:$PATH \
    PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH

### NIXL SETUP ###
# Clone nixl source with shallow clone for faster download
RUN git clone --depth 1 --branch ${NIXL_REF} "https://github.com/ai-dynamo/nixl.git" ${NIXL_SRC_DIR} && \
    cd ${NIXL_SRC_DIR} && \
    if [ "$ARCH" = "arm64" ]; then \
        nixl_build_args="-Ddisable_gds_backend=true"; \
    else \
        nixl_build_args=""; \
    fi && \
    meson setup build/ --buildtype=release --prefix=$NIXL_PREFIX $nixl_build_args && \
    ninja -C build/ -j$(nproc) && \
    ninja -C build/ install && \
    echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Install NIXL Python module
# TODO OPS-590: Move gds_path selection based on arch into NIXL build and re-enable gds backend for arm64
RUN if [ "$ARCH" = "arm64" ]; then \
        cd ${NIXL_SRC_DIR} && uv build . --out-dir /opt/dynamo/wheelhouse/nixl \
        --config-settings=setup-args="-Ddisable_gds_backend=true"; \
    else \
        cd ${NIXL_SRC_DIR} && uv build . --out-dir /opt/dynamo/wheelhouse/nixl; \
    fi

# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python 3.12

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

# Install common and test dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    uv pip install --requirement /tmp/requirements.txt --requirement /tmp/requirements.test.txt

##################################
##### Wheel Build Image ##########
##################################

# Redeclare ARCH_ALT ARG so it's available for interpolation in the FROM instruction
ARG ARCH_ALT

FROM quay.io/pypa/manylinux_2_28_${ARCH_ALT} AS wheel_builder

ARG CARGO_BUILD_JOBS
# Set CARGO_BUILD_JOBS to 16 if not provided
# This is to prevent cargo from building $(nproc) jobs in parallel,
# which might exceed the number of opened files limit.
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16}
# Use build arg RELEASE_BUILD = true to generate wheels for Python 3.10, 3.11 and 3.12.
ARG RELEASE_BUILD
# Use arg ENABLE_KVBM = true to turn on the block-manager feature
ARG ENABLE_KVBM

WORKDIR /opt/dynamo

RUN dnf update -y \
    && dnf install -y llvm-toolset protobuf-compiler python3.12-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    VIRTUAL_ENV=/opt/dynamo/venv \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl

COPY --from=base $RUSTUP_HOME $RUSTUP_HOME
COPY --from=base $CARGO_HOME $CARGO_HOME
COPY --from=base $NIXL_PREFIX $NIXL_PREFIX
COPY --from=base $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH=$CARGO_HOME/bin:$VIRTUAL_ENV/bin:$PATH

# Copy configuration files first for better layer caching
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/

# Copy source code
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build dynamo wheel
RUN uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    uv pip install maturin[patchelf] && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        maturin build --release --features block-manager --out /opt/dynamo/dist; \
    else \
        maturin build --release --out /opt/dynamo/dist; \
    fi && \
    if [ "$RELEASE_BUILD" = "true" ]; then \
        # do not enable KVBM feature, ensure compatibility with lower glibc
        uv run --python 3.11 maturin build --release --out /opt/dynamo/dist && \
        uv run --python 3.10 maturin build --release --out /opt/dynamo/dist; \
    fi

##############################################
########## Dev entrypoint image ##############
##############################################
FROM base AS dev

# Application environment variables
ENV DYNAMO_HOME=/opt/dynamo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    PYTHONPATH=/opt/dynamo:$PYTHONPATH

WORKDIR /opt/dynamo

COPY --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
COPY --from=wheel_builder $CARGO_TARGET_DIR $CARGO_TARGET_DIR

# Copy Cargo cache to avoid re-downloading dependencies
COPY --from=wheel_builder $CARGO_HOME $CARGO_HOME

# Temporarily copy benchmarks folder for installation
COPY benchmarks/ /opt/dynamo/benchmarks/

# Install all python packages
RUN uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*cp312*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl \
    /opt/dynamo/benchmarks && \
    rm -rf /opt/dynamo/benchmarks

# Copy launch banner
RUN --mount=type=bind,source=./container/launch_message.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > ~/.launch_screen && \
    echo "cat ~/.launch_screen" >> ~/.bashrc

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
