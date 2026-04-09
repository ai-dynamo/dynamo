{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/trtllm_runtime.Dockerfile ===
##################################################
########## Runtime Image ########################
##################################################
#
# PURPOSE: Production runtime environment
#
# This stage creates a lightweight production-ready image containing:
# - Pre-compiled TensorRT-LLM and framework dependencies
# - Dynamo runtime libraries and Python packages
# - Essential runtime dependencies and configurations
# - Optimized for inference workloads and deployment
#
# Use this stage when you need:
# - Production deployment of Dynamo with TensorRT-LLM
# - Minimal runtime footprint without build tools
# - Ready-to-run inference server environment
# - Base for custom application containers
#

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG TARGETARCH
WORKDIR /workspace
ENV ENV=${ENV:-/etc/shinit_v2}
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
# workaround for pickle lib issue
ENV OMPI_MCA_coll_ucc_enable=0

# Copy CUDA development tools (nvcc, headers, dependencies, etc.) from PyTorch base image
COPY --from=pytorch_base /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
COPY --from=pytorch_base /usr/local/cuda/bin/nvlink /usr/local/cuda/bin/nvlink
COPY --from=pytorch_base /usr/local/cuda/bin/cudafe++ /usr/local/cuda/bin/cudafe++
COPY --from=pytorch_base /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=pytorch_base /usr/local/cuda/bin/fatbinary /usr/local/cuda/bin/fatbinary
COPY --from=pytorch_base /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=pytorch_base /usr/local/cuda/nvvm /usr/local/cuda/nvvm
COPY --from=pytorch_base /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=pytorch_base /usr/local/cuda/lib64/libcupti* /usr/local/cuda/lib64/
COPY --from=pytorch_base /usr/local/lib/lib* /usr/local/lib/
COPY --from=pytorch_base /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/cuobjdump
COPY --from=pytorch_base /usr/local/cuda/bin/nvdisasm /usr/local/cuda/bin/nvdisasm

ENV CUDA_HOME=/usr/local/cuda \
    TRITON_CUPTI_PATH=/usr/local/cuda/include \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUDART_PATH=/usr/local/cuda/include

# Copy OpenMPI from PyTorch base image
COPY --from=pytorch_base /opt/hpcx/ompi /opt/hpcx/ompi
# Copy NUMA library from PyTorch base image (arch-dependent path)
RUN --mount=type=bind,from=pytorch_base,source=/usr/lib,target=/mnt/usr_lib \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    mkdir -p /usr/lib/${ARCH_ALT}-linux-gnu && \
    cp /mnt/usr_lib/${ARCH_ALT}-linux-gnu/libnuma.so* /usr/lib/${ARCH_ALT}-linux-gnu/

# Copy UCX libraries, libucc.so is needed by pytorch. May not need to copy whole hpcx dir but only /opt/hpcx/ucc/
COPY --from=pytorch_base /opt/hpcx /opt/hpcx
# This is needed to make libucc.so visible so pytorch can use it.
ENV LD_LIBRARY_PATH="/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"
# Might not need to copy cusparseLt in the future once it's included in DLFW cuda container
# networkx, packaging, setuptools get overridden by trtllm installation, so not copying them
# pytorch-triton is copied after trtllm installation.
COPY --from=pytorch_base /usr/local/cuda/lib64/libcusparseLt* /usr/local/cuda/lib64/

# Copy nats and etcd from dynamo_base image
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

# Install Python, build-essential and python3-dev as apt dependencies
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
ARG PYTHON_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64"); \
    if [ ${ARCH_ALT} = "x86_64" ]; then \
        ARCH_FOR_GPG=${ARCH_ALT}; \
    else \
        ARCH_FOR_GPG="sbsa"; \
    fi && \
    curl -fsSL --retry 5 --retry-delay 3 \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH_FOR_GPG}/cuda-archive-keyring.gpg \
        -o /usr/share/keyrings/cuda-archive-keyring.gpg &&\
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH_FOR_GPG} /" \
        | tee /etc/apt/sources.list.d/cuda.repo.list > /dev/null &&\
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Build tools
        build-essential \
        g++ \
        ninja-build \
        git \
        git-lfs \
        # required for verification of GPG keys
        gnupg2 \
        # Python runtime - CRITICAL for virtual environment to work
        python${PYTHON_VERSION}-dev \
        python3-pip \
        # jq for polling various endpoints and health checks
        jq \
        # CUDA/ML libraries
        libcudnn9-cuda-13 \
        libnvshmem3-cuda-13 \
        # Network and communication libraries
        libzmq3-dev \
        # RDMA/UCX libraries required to find RDMA devices
        ibverbs-providers \
        ibverbs-utils \
        libibumad3 \
        libibverbs1 \
        libnuma1 \
        numactl \
        librdmacm1 \
        rdma-core \
        # OpenMPI dependencies
        openssh-client \
        openssh-server \
        # System utilities and dependencies
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create libnccl.so symlink pointing to libnccl.so.2. TensorRT-LLM requires explicit libnccl.so
    ln -sf /usr/lib/${ARCH_ALT}-linux-gnu/libnccl.so.2 /usr/lib/${ARCH_ALT}-linux-gnu/libnccl.so

# nvcr.io/nvidia/cuda-dl-base includes the AWS OFI NCCL plugin, which can crash TRTLLM.
# Disable it by renaming aws-ofi-nccl.conf and refreshing the dynamic linker cache.
RUN if [ -f /etc/ld.so.conf.d/aws-ofi-nccl.conf ]; then \
      mv /etc/ld.so.conf.d/aws-ofi-nccl.conf /etc/ld.so.conf.d/aws-ofi-nccl.conf.disabled; \
    fi && \
    ldconfig

{% if context.trtllm.enable_media_ffmpeg == "true" %}
# Copy ffmpeg libraries from wheel_builder (requires root, runs before USER dynamo)
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

# Copy TensorRT and libgomp from framework image (arch-dependent path, needs root)
COPY --from=framework /usr/local/tensorrt /usr/local/tensorrt
RUN --mount=type=bind,from=framework,source=/usr/lib,target=/mnt/usr_lib \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    cp /mnt/usr_lib/${ARCH_ALT}-linux-gnu/libgomp.so* /usr/lib/${ARCH_ALT}-linux-gnu/

# Register arch-dependent TensorRT and nvshmem library paths with ldconfig so the
# dynamic linker finds them in every execution context (docker run, exec, k8s, etc.)
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    echo "/usr/local/tensorrt/targets/${ARCH_ALT}-linux-gnu/lib" > /etc/ld.so.conf.d/tensorrt.conf && \
    echo "/usr/lib/${ARCH_ALT}-linux-gnu/nvshmem/13" >> /etc/ld.so.conf.d/tensorrt.conf && \
    ldconfig

# Switch to dynamo user
USER dynamo
ENV HOME=/home/dynamo
# This picks up the umask 002 from the /etc/profile.d/00-umask.sh file for subsequent RUN commands
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]

ENV DYNAMO_HOME=/workspace
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib64
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

# Copy pre-built venv with PyTorch and TensorRT-LLM from framework stage
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 --from=framework ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy UCX from framework image as plugin for NIXL
# Copy NIXL source from framework image
# Copy dynamo wheels for gitlab artifacts (read-only, no group-write needed)
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo: --from=wheel_builder $NIXL_PREFIX $NIXL_PREFIX
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

ENV PATH="/usr/local/ucx/bin:${VIRTUAL_ENV}/bin:/opt/hpcx/ompi/bin:/usr/local/bin/etcd/:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:$PATH"
# Both arch paths are listed; the non-existent one is silently ignored by the linker.
ENV LD_LIBRARY_PATH=\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
/opt/hpcx/ompi/lib:\
/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:\
/usr/local/tensorrt/targets/aarch64-linux-gnu/lib:\
/usr/lib/x86_64-linux-gnu/nvshmem/13/:\
/usr/lib/aarch64-linux-gnu/nvshmem/13/:\
/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:\
/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch_tensorrt/lib:\
/usr/local/cuda/lib:\
/usr/local/cuda/lib64:\
$LD_LIBRARY_PATH
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENV OPAL_PREFIX=/opt/hpcx/ompi

# TODO: skip /workspace COPYs for dev/local-dev (bind-mounted from host, these get shadowed)
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/
{% if target not in ("dev", "local-dev") %}
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/
{% endif %}

# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if target not in ("dev", "local-dev") %}
# Install dynamo, NIXL, and dynamo-specific dependencies
ARG ENABLE_KVBM
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
      /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
      /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
      /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM is true but no KVBM wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$KVBM_WHEEL"; \
    fi && \
    cd /workspace/benchmarks && \
    UV_GIT_LFS=1 uv pip install --no-cache . && \
    chmod -R g+w /workspace/benchmarks
{% else %}
# Dev/local-dev: skip dynamo wheel install (users build from source via cargo build + maturin develop).
# Install NIXL wheel only (pre-built C++ binary, not buildable from source).
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

# Install gpu_memory_service wheel if enabled (all targets)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install "$GMS_WHEEL"; fi; \
    fi

# Optional: ModelExpress P2P weight transfer support
ARG ENABLE_MODELEXPRESS_P2P
ARG MODELEXPRESS_REF
RUN if [ "${ENABLE_MODELEXPRESS_P2P}" = "true" ]; then \
        echo "Installing ModelExpress P2P support..."; \
        uv pip install "modelexpress @ git+https://github.com/ai-dynamo/modelexpress.git@${MODELEXPRESS_REF}#subdirectory=modelexpress_client/python"; \
    fi

# Install runtime dependencies (common + benchmarks).
# Test and dev dependencies are NOT installed here — they go in the test and dev images.
# --no-cache is intentional: mixed indexes (PyPI + PyTorch CUDA wheels) risk serving stale/wrong-variant cached wheels
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.benchmark.txt,target=/tmp/requirements.benchmark.txt \
    export UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --no-cache \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.common.txt \
        --requirement /tmp/requirements.benchmark.txt \
        cupy-cuda13x && \
    # nvidia-cutlass-dsl-libs-base==4.4.1 (transitive dep) ships a stub cute/experimental/__init__.py
    # that unconditionally raises NotImplementedError, crashing TRT-LLM on import. cutlass-dsl==4.3.4
    # (pinned by TRT-LLM) works without cute/experimental/. Remove the stub to fix the NotImplementedError.
    rm -rf ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/experimental/

# Copy tests, deploy, and the trtllm/common/mocker component subtrees for CI.
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/trtllm /workspace/components/src/dynamo/trtllm
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Setup environment for all users
USER root
# Fix directory permissions: COPY --chmod only affects contents, not the directory itself
RUN chmod g+w ${VIRTUAL_ENV} /workspace /workspace/* /opt/dynamo /opt/dynamo/* && \
    chown dynamo:0 ${VIRTUAL_ENV} /workspace /opt/dynamo/ && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
