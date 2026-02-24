{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/vllm_runtime.Dockerfile ===
##################################################
########## Runtime Image ########################
##################################################
#
# PURPOSE: Production runtime environment
#
# This stage creates a lightweight production-ready image containing:
# - Pre-compiled vLLM and framework dependencies
# - Dynamo runtime libraries and Python packages
# - Essential runtime dependencies and configurations
# - Optimized for inference workloads and deployment
#
# Use this stage when you need:
# - Production deployment of Dynamo with vLLM
# - Minimal runtime footprint without build tools
# - Ready-to-run inference server environment
# - Base for custom application containers
#

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

WORKDIR /workspace
ENV DYNAMO_HOME=/opt/dynamo
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
# Set CUDA_DEVICE_ORDER to ensure CUDA logical device IDs match NVML physical device IDs
# This fixes NVML InvalidArgument errors when CUDA_VISIBLE_DEVICES is set
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Copy CUDA development tools (nvcc, headers, dependencies, etc.) from base devel image
COPY --from=dynamo_base /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
COPY --from=dynamo_base /usr/local/cuda/bin/nvlink /usr/local/cuda/bin/nvlink
COPY --from=dynamo_base /usr/local/cuda/bin/cudafe++ /usr/local/cuda/bin/cudafe++
COPY --from=dynamo_base /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=dynamo_base /usr/local/cuda/bin/fatbinary /usr/local/cuda/bin/fatbinary
COPY --from=dynamo_base /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=dynamo_base /usr/local/cuda/nvvm /usr/local/cuda/nvvm
COPY --from=dynamo_base /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=dynamo_base /usr/local/cuda/lib64/stubs/ /usr/local/cuda/lib64/stubs/
RUN CUDA_VERSION_MAJOR="${CUDA_VERSION%%.*}" &&\
    ln -s /usr/local/cuda/lib64/libcublas.so.${CUDA_VERSION_MAJOR} /usr/local/cuda/lib64/libcublas.so &&\
    ln -s /usr/local/cuda/lib64/libcublasLt.so.${CUDA_VERSION_MAJOR} /usr/local/cuda/lib64/libcublasLt.so

# DeepGemm runs nvcc for JIT kernel compilation, however the CUDA include path
# is not properly set for complilation. Set CPATH to help nvcc find the headers.
ENV CPATH=/usr/local/cuda/include

### COPY NATS & ETCD ###
# Copy nats and etcd from dev image
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
# Add ETCD and CUDA binaries to PATH so cicc and other CUDA tools are accessible
ENV PATH=/usr/local/bin/etcd/:/usr/local/cuda/nvvm/bin:$PATH

# Copy uv to system /bin
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

ARG ARCH_ALT
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Install Python, build-essential and python3-dev as apt dependencies
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    CUDA_VERSION_MAJOR=${CUDA_VERSION%%.*} &&\
    CUDA_VERSION_MINOR=$(echo "${CUDA_VERSION#*.}" | cut -d. -f1) && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Python runtime - CRITICAL for virtual environment to work
        python${PYTHON_VERSION}-dev \
        build-essential \
        # jq and curl for polling various endpoints and health checks
        jq \
        git \
        git-lfs \
        # required for verification of GPG keys
        gnupg2 \
        curl \
        # Libraries required by UCX to find RDMA devices
        libibverbs1 rdma-core ibverbs-utils libibumad3 \
        libnuma1 librdmacm1 ibverbs-providers \
        # JIT Kernel Compilation, flashinfer
        ninja-build \
        g++ \
        # prometheus dependencies
        ca-certificates \
        # opencv-python-headless (vLLM dependency) requires libxcb for some functions
        libxcb1 \
        # DeepGemm uses 'cuobjdump' which does not come with CUDA image
        cuda-command-line-tools-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} && \
    rm -rf /var/lib/apt/lists/*

{% if context.vllm.enable_media_ffmpeg == "true" %}
# Copy ffmpeg libraries from wheel_builder (requires root, runs before USER dynamo)
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

USER dynamo
ENV HOME=/home/dynamo
# This picks up the umask 002 from the /etc/profile.d/00-umask.sh file for subsequent RUN commands
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]

ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/${ARCH_ALT}-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

# Site-packages path derived from PYTHON_VERSION ARG
ARG SITE_PACKAGES=${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages

### VIRTUAL ENVIRONMENT SETUP ###
# Copy virtual environment from framework container, splitting large packages into separate layers
# to enable parallel downloads. Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as
# root because COPY --chmod only affects <path>/*, not <path>
#
# Layer sizes (uncompressed): nvidia=4.5GB, flashinfer_jit_cache=4.1GB, torch=2.1GB,
#                             vllm=1.2GB, triton=592MB, flashinfer_cubin=437MB
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/nvidia ${SITE_PACKAGES}/nvidia
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/flashinfer_jit_cache ${SITE_PACKAGES}/flashinfer_jit_cache
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/torch ${SITE_PACKAGES}/torch
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/vllm ${SITE_PACKAGES}/vllm
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/triton ${SITE_PACKAGES}/triton
COPY --chmod=775 --chown=dynamo:0 --from=framework ${SITE_PACKAGES}/flashinfer_cubin ${SITE_PACKAGES}/flashinfer_cubin
# Remaining packages and venv structure (bin/, include/, share/, etc.)
COPY --chmod=775 --chown=dynamo:0 --from=framework \
    --exclude=lib/python*/site-packages/nvidia \
    --exclude=lib/python*/site-packages/flashinfer_jit_cache \
    --exclude=lib/python*/site-packages/torch \
    --exclude=lib/python*/site-packages/vllm \
    --exclude=lib/python*/site-packages/triton \
    --exclude=lib/python*/site-packages/flashinfer_cubin \
    ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy vllm with correct ownership (read-only, no group-write needed)
COPY --chown=dynamo:0 --from=framework /opt/vllm /opt/vllm

# Copy UCX and NIXL to system directories (read-only, no group-write needed)
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo: --from=wheel_builder $NIXL_PREFIX $NIXL_PREFIX
COPY --chown=dynamo: --from=wheel_builder /opt/nvidia/nvda_nixl/lib64/. ${NIXL_LIB_DIR}/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# Copy AWS SDK C++ libraries (required for NIXL OBJ backend / S3 support)
COPY --chown=dynamo: --from=wheel_builder /usr/local/lib64/libaws* /usr/local/lib/
COPY --chown=dynamo: --from=wheel_builder /usr/local/lib64/libs2n* /usr/local/lib/
COPY --chown=dynamo: --from=wheel_builder /usr/lib64/libcrypto.so.1.1* /usr/local/lib/
COPY --chown=dynamo: --from=wheel_builder /usr/lib64/libssl.so.1.1* /usr/local/lib/

ENV PATH=/usr/local/ucx/bin:$PATH

ENV LD_LIBRARY_PATH=\
/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
$LD_LIBRARY_PATH
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Copy attribution files
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/

# Install dynamo, NIXL, and dynamo-specific dependencies
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
      /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
      /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
      /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -z "$GMS_WHEEL" ]; then \
            echo "ERROR: ENABLE_GPU_MEMORY_SERVICE is true but no gpu_memory_service wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$GMS_WHEEL"; \
    fi && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM is true but no KVBM wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$KVBM_WHEEL"; \
    fi && \
    cd /workspace/benchmarks && \
    export UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install . && \
    # pip/uv bypasses umask when creating .egg-info files, but chmod -R is fast here (small directory)
    chmod -R g+w /workspace/benchmarks


# Install ModelExpress for P2P weight transfer (optional)
ARG ENABLE_MODELEXPRESS_P2P
ARG MODELEXPRESS_REF
RUN if [ "${ENABLE_MODELEXPRESS_P2P}" = "true" ]; then \
        echo "Installing ModelExpress from ref: ${MODELEXPRESS_REF}" && \
        uv pip install "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git@${MODELEXPRESS_REF}#subdirectory=modelexpress_client/python"; \
    fi

# Install common and test dependencies. Cache uv downloads; uv handles its own locking for this cache.
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt

# Copy tests, deploy and components for CI with correct ownership
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
COPY --chmod=775 --chown=dynamo:0 components/ /workspace/components/
COPY --chmod=775 --chown=dynamo:0 lib/ /workspace/lib/

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Setup environment for all users
USER root
# Fix directory permissions: COPY --chmod only affects contents, not the directory itself
RUN chmod g+w /workspace /workspace/* /opt/dynamo /opt/dynamo/* ${VIRTUAL_ENV} && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

# Fix library symlinks that Docker COPY dereferenced (COPY always follows symlinks)
# This recreates proper symlinks to save space and suppress ldconfig warnings
RUN cd /usr/local/lib && \
    # libaws-c-common: .so.1 should symlink to .so.1.0.0
    if [ -f libaws-c-common.so.1.0.0 ] && [ ! -L libaws-c-common.so.1 ]; then \
        rm -f libaws-c-common.so.1 libaws-c-common.so && \
        ln -s libaws-c-common.so.1.0.0 libaws-c-common.so.1 && \
        ln -s libaws-c-common.so.1 libaws-c-common.so; \
    fi && \
    # libaws-c-s3: .so.0unstable should symlink to .so.1.0.0
    if [ -f libaws-c-s3.so.1.0.0 ] && [ ! -L libaws-c-s3.so.0unstable ]; then \
        rm -f libaws-c-s3.so.0unstable libaws-c-s3.so && \
        ln -s libaws-c-s3.so.1.0.0 libaws-c-s3.so.0unstable && \
        ln -s libaws-c-s3.so.0unstable libaws-c-s3.so; \
    fi && \
    # libs2n: .so.1 should symlink to .so.1.0.0
    if [ -f libs2n.so.1.0.0 ] && [ ! -L libs2n.so.1 ]; then \
        rm -f libs2n.so.1 libs2n.so && \
        ln -s libs2n.so.1.0.0 libs2n.so.1 && \
        ln -s libs2n.so.1 libs2n.so; \
    fi && \
    # OpenSSL 1.1: check for versioned files (e.g., .so.1.1.1k)
    for lib in libcrypto libssl; do \
        versioned=$(ls -1 ${lib}.so.1.1.* 2>/dev/null | head -1); \
        if [ -n "$versioned" ] && [ ! -L "${lib}.so.1.1" ]; then \
            rm -f "${lib}.so.1.1" && \
            ln -s "$(basename "$versioned")" "${lib}.so.1.1"; \
        fi; \
    done && \
    ldconfig

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

# In vLLM 0.12 the default sampler changed on the forward pass.
# We need to enable this to enable the cuda kernels.
ENV VLLM_USE_FLASHINFER_SAMPLER=1

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
