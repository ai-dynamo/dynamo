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

ARG ARCH_ALT
ARG PYTHON_VERSION
ARG ENABLE_KVBM
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/${ARCH_ALT}-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

# Install Python, build-essential and python3-dev as apt dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Python runtime - CRITICAL for virtual environment to work
        python${PYTHON_VERSION}-dev \
        build-essential \
        # jq and curl for polling various endpoints and health checks
        jq \
        git \
        git-lfs \
        curl \
        # Libraries required by UCX to find RDMA devices
        libibverbs1 rdma-core ibverbs-utils libibumad3 \
        libnuma1 librdmacm1 ibverbs-providers \
        # JIT Kernel Compilation, flashinfer
        ninja-build \
        g++ \
        # prometheus dependencies
        ca-certificates \
        # DeepGemm uses 'cuobjdump' which does not come with CUDA image
        cuda-command-line-tools-12-8 && \
    rm -rf /var/lib/apt/lists/*

# Copy CUDA development tools (nvcc, headers, dependencies, etc.) from base devel image
COPY --from=framework /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
COPY --from=framework /usr/local/cuda/bin/cudafe++ /usr/local/cuda/bin/cudafe++
COPY --from=framework /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=framework /usr/local/cuda/bin/fatbinary /usr/local/cuda/bin/fatbinary
COPY --from=framework /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=framework /usr/local/cuda/nvvm /usr/local/cuda/nvvm
COPY --from=framework /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=framework /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=framework /usr/local/cuda/lib64/libcublasLt.so* /usr/local/cuda/lib64/

### COPY NATS & ETCD ###
# Copy nats and etcd from dev image
COPY --from=dynamo_dev /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_dev /usr/local/bin/etcd/ /usr/local/bin/etcd/
# Add ETCD and CUDA binaries to PATH so cicc and other CUDA tools are accessible
ENV PATH=/usr/local/bin/etcd/:/usr/local/cuda/nvvm/bin:$PATH

# DeepGemm runs nvcc for JIT kernel compilation, however the CUDA include path
# is not properly set for complilation. Set CPATH to help nvcc find the headers.
ENV CPATH=/usr/local/cuda/include

# Copy uv to system /bin
COPY --from=framework /bin/uv /bin/uvx /bin/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && chown -R dynamo: /workspace /home/dynamo /opt/dynamo \
    && chmod -R g+w /workspace /home/dynamo/.cache /opt/dynamo

USER dynamo
ENV HOME=/home/dynamo

# Copy UCX and NIXL to system directories
COPY --chown=dynamo: --from=dynamo_dev /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo: --from=dynamo_dev $NIXL_PREFIX $NIXL_PREFIX
ENV PATH=/usr/local/ucx/bin:$PATH

ENV LD_LIBRARY_PATH=\
/opt/vllm/tools/ep_kernels/ep_kernels_workspace/nvshmem_install/lib:\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
$LD_LIBRARY_PATH

### VIRTUAL ENVIRONMENT SETUP ###

# Copy entire virtual environment from framework container with correct ownership
COPY --chown=dynamo: --from=framework ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy vllm with correct ownership
COPY --chown=dynamo: --from=framework /opt/vllm /opt/vllm

# Install dynamo, NIXL, and dynamo-specific dependencies
COPY --chown=dynamo: benchmarks/ /opt/dynamo/benchmarks/
COPY --chown=dynamo: --from=dynamo_dev /opt/dynamo/wheelhouse/ /opt/dynamo/wheelhouse/
RUN uv pip install \
      /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
      /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
      /opt/dynamo/wheelhouse/nixl/nixl*.whl \
    && if [ "${ENABLE_KVBM}" = "true" ]; then \
        uv pip install /opt/dynamo/wheelhouse/kvbm*.whl; \
       fi \
    && cd /opt/dynamo/benchmarks \
    && UV_GIT_LFS=1 uv pip install --no-cache . \
    && cd - \
    && rm -rf /opt/dynamo/benchmarks

# Install common and test dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    UV_GIT_LFS=1 uv pip install \
        --no-cache \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt

# Copy benchmarks, examples, and tests for CI with correct ownership
COPY --chown=dynamo: . /workspace/

# Copy attribution files
COPY --chown=dynamo: ATTRIBUTION* LICENSE /workspace/

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Setup environment for all users
USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
