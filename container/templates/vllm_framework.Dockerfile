########################################################
########## Framework Development Image ################
########################################################
#
# PURPOSE: Framework development and vLLM compilation
#
# This stage builds and compiles framework dependencies including:
# - vLLM inference engine with CUDA support
# - DeepGEMM and FlashInfer optimizations
# - All necessary build tools and compilation dependencies
# - Framework-level Python packages and extensions
#
# Use this stage when you need to:
# - Build vLLM from source with custom modifications
# - Develop or debug framework-level components
# - Create custom builds with specific optimization flags
#

# Use dynamo base image (see /container/Dockerfile for more details)
FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

ARG PYTHON_VERSION

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Python runtime - CRITICAL for virtual environment to work
        python${PYTHON_VERSION}-dev \
        build-essential \
        # vLLM build dependencies
        cmake \
        ibverbs-providers \
        ibverbs-utils \
        libibumad-dev \
        libibverbs-dev \
        libnuma-dev \
        librdmacm-dev \
        rdma-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# if libmlx5.so not shipped with 24.04 rdma-core packaging, CMAKE will fail when looking for
# generic dev name .so so we symlink .s0.1 -> .so
RUN ln -sf /usr/lib/aarch64-linux-gnu/libmlx5.so.1 /usr/lib/aarch64-linux-gnu/libmlx5.so || true

# Create virtual environment
RUN mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

ARG ARCH
# Install vllm - keep this early in Dockerfile to avoid
# rebuilds from unrelated source code changes
ARG VLLM_REF
ARG DEEPGEMM_REF
ARG FLASHINF_REF
ARG LMCACHE_REF
ARG CUDA_VERSION

ARG MAX_JOBS
ENV MAX_JOBS=$MAX_JOBS
ENV CUDA_HOME=/usr/local/cuda

# Install VLLM and related dependencies
RUN --mount=type=bind,source=./container/deps/,target=/tmp/deps \
    --mount=type=cache,target=/root/.cache/uv \
    cp /tmp/deps/vllm/install_vllm.sh /tmp/install_vllm.sh && \
    chmod +x /tmp/install_vllm.sh && \
    /tmp/install_vllm.sh \
        --vllm-ref $VLLM_REF \
        --max-jobs $MAX_JOBS \
        --arch $ARCH \
        --installation-dir /opt \
        ${DEEPGEMM_REF:+--deepgemm-ref "$DEEPGEMM_REF"} \
        ${FLASHINF_REF:+--flashinf-ref "$FLASHINF_REF"} \
        ${LMCACHE_REF:+--lmcache-ref "$LMCACHE_REF"} \
        --cuda-version $CUDA_VERSION

ENV LD_LIBRARY_PATH=\
/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:\
$LD_LIBRARY_PATH
