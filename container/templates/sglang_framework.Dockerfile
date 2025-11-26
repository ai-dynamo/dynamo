ARG CUDA_VERSION=12.9.1

# Runtime image and build-time configuration (aligned with other backends)
# TODO: OPS-<number>: Use the same runtime image as the other backends
ARG RUNTIME_IMAGE="nvcr.io/nvidia/cuda"
ARG RUNTIME_IMAGE_TAG="12.9.1-cudnn-runtime-ubuntu24.04"

ARG PYTHON_VERSION=3.10
ARG ARCH=amd64
ARG ARCH_ALT=x86_64
ARG CARGO_BUILD_JOBS

# sccache configuration - inherit from base build
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

########################################################
########## Framework Development Image ################
########################################################
#
# PURPOSE: Framework development and SGLang/DeepEP/NVSHMEM compilation
#
# This stage builds and compiles framework dependencies including:
# - SGLang inference engine with CUDA support
# - DeepEP and NVSHMEM
# - All necessary build tools and compilation dependencies
# - Framework-level Python packages and extensions
#
# Use this stage when you need to:
# - Build SGLang from source with custom modifications
# - Develop or debug framework-level components
# - Create custom builds with specific optimization flags
#
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS framework

# Declare all ARGs
ARG BUILD_TYPE=all
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG DEEPEP_GB_COMMIT=1b14ad661c7640137fcfe93cccb2694ede1220b0
ARG CMAKE_BUILD_PARALLEL_LEVEL=2
ARG SGL_KERNEL_VERSION=0.3.16.post5
ARG SGLANG_COMMIT=0.5.4.post3
ARG GDRCOPY_COMMIT=v2.4.4
ARG NVSHMEM_VERSION=3.3.9
ARG GRACE_BLACKWELL=false
ARG ARCH
ARG ARCH_ALT
ARG PYTHON_VERSION
ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
ARG CARGO_BUILD_JOBS
ARG CUDA_VERSION

# Set all environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install \
    PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Combined: Python setup, locale, and all package installation
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Python (using other python versions as needed)
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        # Build essentials
        build-essential \
        cmake \
        ninja-build \
        ccache \
        patchelf \
        git \
        git-lfs \
        # Core system utilities
        tzdata \
        locales \
        ca-certificates \
        dkms \
        kmod \
        # Command line tools
        wget \
        curl \
        jq \
        unzip \
        # Network utilities
        netcat-openbsd \
        # SSL and pkg-config
        libssl-dev \
        pkg-config \
        # MPI and NUMA
        libopenmpi-dev \
        libnuma1 \
        libnuma-dev \
        numactl \
        # InfiniBand/RDMA
        libibverbs-dev \
        libibverbs1 \
        libibumad3 \
        librdmacm1 \
        libnl-3-200 \
        libnl-route-3-200 \
        libnl-route-3-dev \
        libnl-3-dev \
        ibverbs-providers \
        infiniband-diags \
        perftest \
        # Development libraries
        libgoogle-glog-dev \
        libgtest-dev \
        libjsoncpp-dev \
        libunwind-dev \
        libboost-all-dev \
        libgrpc-dev \
        libgrpc++-dev \
        libprotobuf-dev \
        protobuf-compiler \
        protobuf-compiler-grpc \
        pybind11-dev \
        libhiredis-dev \
        libcurl4-openssl-dev \
        libczmq4 \
        libczmq-dev \
        libfabric-dev \
        # Package building tools
        devscripts \
        debhelper \
        fakeroot \
        check \
        libsubunit0 \
        libsubunit-dev \
    # Set Python alternatives
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python /usr/bin/python${PYTHON_VERSION} \
    # Set up locale
    && locale-gen en_US.UTF-8 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install sccache if requested
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
    /tmp/use-sccache.sh install; \
fi

# Set environment variables - they'll be empty strings if USE_SCCACHE=false
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}} \
    SCCACHE_S3_KEY_PREFIX=${USE_SCCACHE:+${ARCH}} \
    RUSTC_WRAPPER=${USE_SCCACHE:+sccache} \
    CMAKE_C_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CXX_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache} \
    CMAKE_CUDA_COMPILER_LAUNCHER=${USE_SCCACHE:+sccache}

WORKDIR /sgl-workspace

# GDRCopy installation
RUN git clone --depth 1 --branch ${GDRCOPY_COMMIT} https://github.com/NVIDIA/gdrcopy.git \
    && cd gdrcopy/packages \
    && export CUDA=${CUDA_HOME} \
    && ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Create dynamo user EARLY - before copying files, with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /workspace /home/dynamo/.cache /opt/dynamo \
    && chown -R dynamo: /sgl-workspace /workspace /home/dynamo /opt/dynamo \
    && chmod -R g+w /sgl-workspace /workspace /home/dynamo/.cache /opt/dynamo

USER dynamo
ENV HOME=/home/dynamo

# Install SGLang (requires CUDA 12.8.1 or 12.9.1)
RUN python3 -m pip install --no-cache-dir --ignore-installed pip==25.3 setuptools==80.9.0 wheel==0.45.1 html5lib==1.1 six==1.17.0 \
    && git clone --depth 1 --branch v${SGLANG_COMMIT} https://github.com/sgl-project/sglang.git \
    && cd sglang \
    && case "$CUDA_VERSION" in \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        *) echo "Error: Unsupported CUDA version for sglang: $CUDA_VERSION (requires 12.8.1 or 12.9.1)" && exit 1 ;; \
    esac \
    && python3 -m pip install --no-cache-dir sgl-kernel==${SGL_KERNEL_VERSION} \
    && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
    && python3 -m pip install --no-cache-dir nvidia-nccl-cu12==2.27.6 --force-reinstall --no-deps \
    && FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin

# Download and extract NVSHMEM source, clone DeepEP (use Tom's fork for GB200)
RUN --mount=type=cache,target=/var/cache/curl,uid=1000,gid=0 \
    curl --retry 3 --retry-delay 2 -fsSL -o /var/cache/curl/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && tar -xf /var/cache/curl/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && mv nvshmem_src nvshmem \
    && rm -f /var/cache/curl/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && if [ "$GRACE_BLACKWELL" = true ]; then \
        git clone --depth 1 https://github.com/fzyzcjy/DeepEP.git \
        && cd DeepEP \
        && git fetch --depth 1 origin ${DEEPEP_GB_COMMIT} \
        && git checkout ${DEEPEP_GB_COMMIT}; \
    else \
        git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git \
        && cd DeepEP \
        && git fetch --depth 1 origin ${DEEPEP_COMMIT} \
        && git checkout ${DEEPEP_COMMIT}; \
    fi \
    && sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh

# Build and install NVSHMEM library only (without python library)
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    cd /sgl-workspace/nvshmem && \
    if [ "$GRACE_BLACKWELL" = true ]; then CUDA_ARCH="90;100;120"; else CUDA_ARCH="90"; fi && \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DNVSHMEM_BUILD_PYTHON_LIB=OFF && \
    cmake --build build --target install -j${CMAKE_BUILD_PARALLEL_LEVEL} && \
    /tmp/use-sccache.sh show-stats "NVSHMEM"

# Build nvshmem4py wheels separately (Python 3.10, CUDA 12) to avoid building the python library twice for multiple python versions
# Need to reconfigure with PYTHON_LIB=ON to add the nvshmem4py subdirectory
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    cd /sgl-workspace/nvshmem && \
    if [ "$GRACE_BLACKWELL" = true ]; then CUDA_ARCH="90;100;120"; else CUDA_ARCH="90"; fi && \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DNVSHMEM_BUILD_PYTHON_LIB=ON && \
    cmake --build build --target build_nvshmem4py_wheel_cu12_${PYTHON_VERSION} -j${CMAKE_BUILD_PARALLEL_LEVEL} && \
    /tmp/use-sccache.sh show-stats "NVSHMEM4PY"

# Install DeepEP
RUN --mount=type=secret,id=aws-key-id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${ARCH}} && \
    cd /sgl-workspace/DeepEP && \
    NVSHMEM_DIR=${NVSHMEM_DIR} TORCH_CUDA_ARCH_LIST="9.0;10.0" pip install --no-build-isolation .

# Copy rust installation from dev to avoid duplication efforts
COPY --from=dynamo_dev /usr/local/rustup /usr/local/rustup
COPY --from=dynamo_dev /usr/local/cargo /usr/local/cargo

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/workspace/target \
    PATH=/usr/local/cargo/bin:$PATH \
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16}

# Install essential Python build tools
RUN python3 -m pip install --no-cache-dir \
    mooncake-transfer-engine==0.3.6.post1 \
    scikit-build-core==0.11.6 \
    setuptools-rust==1.12.0
