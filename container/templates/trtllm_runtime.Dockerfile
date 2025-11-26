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

FROM ${TRTLLM_RUNTIME_IMAGE}:${TRTLLM_RUNTIME_IMAGE_TAG} AS runtime

ARG ARCH_ALT
ARG ENABLE_KVBM
ARG PYTHON_VERSION

WORKDIR /workspace

ENV ENV=${ENV:-/etc/shinit_v2}
ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/${ARCH_ALT}-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins
# workaround for pickle lib issue
ENV OMPI_MCA_coll_ucc_enable=0
# Use UCX KVCACHE by default
ENV TRTLLM_USE_UCX_KVCACHE=1

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

# Install Python, build-essential and python3-dev as apt dependencies
RUN if [ ${ARCH_ALT} = "x86_64" ]; then \
        ARCH_FOR_GPG=${ARCH_ALT}; \
    else \
        ARCH_FOR_GPG="sbsa"; \
    fi && \
    curl -fsSL \
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
        librdmacm1 \
        rdma-core \
        # OpenMPI dependencies
        openssh-client \
        openssh-server \
        # System utilities and dependencies
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/lib/${ARCH_ALT}-linux-gnu/nvshmem/13/:${LD_LIBRARY_PATH}"

# Copy CUDA development tools (nvcc, headers, dependencies, etc.) from PyTorch base image
COPY --from=pytorch_base /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
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

# Copy nats and etcd from dev image
COPY --from=dynamo_dev /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_dev /usr/local/bin/etcd/ /usr/local/bin/etcd/
# Add ETCD and CUDA binaries to PATH so cicc and other CUDA tools are accessible
ENV PATH=/usr/local/bin/etcd/:/usr/local/cuda/nvvm/bin:$PATH

# Copy OpenMPI from PyTorch base image
COPY --from=pytorch_base /opt/hpcx/ompi /opt/hpcx/ompi
# Copy NUMA library from PyTorch base image
COPY --from=pytorch_base /usr/lib/${ARCH_ALT}-linux-gnu/libnuma.so* /usr/lib/${ARCH_ALT}-linux-gnu/

# Copy UCX libraries, libucc.so is needed by pytorch. May not need to copy whole hpcx dir but only /opt/hpcx/ucc/
COPY --from=pytorch_base /opt/hpcx /opt/hpcx
# This is needed to make libucc.so visible so pytorch can use it.
ENV LD_LIBRARY_PATH="/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"
# Might not need to copy cusparseLt in the future once it's included in DLFW cuda container
COPY --from=pytorch_base /usr/local/cuda/lib64/libcusparseLt* /usr/local/cuda/lib64/

# Copy uv to system /bin
COPY --from=framework /bin/uv /bin/uvx /bin/

# Copy libgomp.so from framework image
COPY --from=framework /usr/local/tensorrt /usr/local/tensorrt
COPY --from=framework /usr/lib/${ARCH_ALT}-linux-gnu/libgomp.so* /usr/lib/${ARCH_ALT}-linux-gnu/

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    && chown -R dynamo: /workspace /home/dynamo /opt/dynamo \
    && chmod -R g+w /workspace /home/dynamo/.cache /opt/dynamo


# Switch to dynamo user
USER dynamo
ENV HOME=/home/dynamo
ENV DYNAMO_HOME=/workspace

# Copy UCX from framework image as plugin for NIXL
# Copy NIXL source from framework image
# Copy dynamo wheels for gitlab artifacts
COPY --chown=dynamo: --from=dynamo_dev /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo: --from=dynamo_dev $NIXL_PREFIX $NIXL_PREFIX

ENV PATH="/usr/local/ucx/bin:${VIRTUAL_ENV}/bin:/opt/hpcx/ompi/bin:/usr/local/bin/etcd/:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:$PATH"
ENV LD_LIBRARY_PATH=\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
/opt/hpcx/ompi/lib:\
$LD_LIBRARY_PATH
ENV OPAL_PREFIX=/opt/hpcx/ompi

# Copy pre-built venv with PyTorch and TensorRT-LLM from framework stage
COPY --chown=dynamo: --from=framework ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ENV TENSORRT_LIB_DIR=/usr/local/tensorrt/targets/${ARCH_ALT}-linux-gnu/lib
ENV LD_LIBRARY_PATH=/opt/dynamo/venv/lib/python3.12/site-packages/torch/lib:/opt/dynamo/venv/lib/python3.12/site-packages/torch_tensorrt/lib:${TENSORRT_LIB_DIR}:${LD_LIBRARY_PATH}

# Install dynamo, NIXL, and dynamo-specific dependencies
COPY --chown=dynamo: benchmarks/ /opt/dynamo/benchmarks/
COPY --chown=dynamo: --from=dynamo_dev /opt/dynamo/wheelhouse/ /opt/dynamo/wheelhouse/
RUN uv pip install \
      --no-cache \
      /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
      /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
      /opt/dynamo/wheelhouse/nixl/nixl*.whl \
    && if [ "${ENABLE_KVBM}" = "true" ]; then \
        uv pip install --no-cache /opt/dynamo/wheelhouse/kvbm*.whl; \
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
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt \
        cupy-cuda13x

# Copy tests, benchmarks, deploy and components for CI with correct ownership
COPY --chown=dynamo: tests /workspace/tests
COPY --chown=dynamo: examples /workspace/examples
COPY --chown=dynamo: benchmarks /workspace/benchmarks
COPY --chown=dynamo: deploy /workspace/deploy
COPY --chown=dynamo: components/ /workspace/components/
COPY --chown=dynamo: recipes/ /workspace/recipes/

# Copy attribution files with correct ownership
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
