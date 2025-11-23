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
ARG NIXL_REF=0.7.1
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
