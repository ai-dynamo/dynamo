##################################################
########## Runtime Image ########################
##################################################
#
# PURPOSE: Production runtime environment
#
# This stage creates a production-ready image containing:
# - Pre-compiled SGLang, DeepEP, and NVSHMEM components
# - Dynamo runtime libraries and Python packages
# - Essential runtime dependencies and configurations
# - Optimized for inference workloads and deployment
#
# Use this stage when you need:
# - Production deployment of Dynamo with SGLang + DeepEP
# - Minimal runtime footprint without build tools
# - Ready-to-run inference server environment
#
FROM framework AS runtime

WORKDIR /workspace

ARG ARCH
ARG ARCH_ALT
ARG PYTHON_VERSION

ENV DYNAMO_HOME=/opt/dynamo
ENV NVSHMEM_DIR=/sgl-workspace/nvshmem/install
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=${NIXL_PREFIX}/lib/${ARCH_ALT}-linux-gnu
ENV NIXL_PLUGIN_DIR=${NIXL_LIB_DIR}/plugins
ENV LD_LIBRARY_PATH=\
${NVSHMEM_DIR}/lib:\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
/usr/local/nvidia/lib64:\
${LD_LIBRARY_PATH}

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA


# Copy NATS and ETCD from dev, and UCX/NIXL
COPY --from=dynamo_dev /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_dev /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_dev /usr/local/ucx /usr/local/ucx
COPY --from=dynamo_dev $NIXL_PREFIX $NIXL_PREFIX
ENV PATH=/usr/local/bin/etcd/:/usr/local/cuda/nvvm/bin:${HOME}/.local/bin:$PATH

# Install Dynamo wheels from dev wheelhouse
COPY --chown=dynamo: benchmarks/ /opt/dynamo/benchmarks/
COPY --chown=dynamo: --from=dynamo_dev /opt/dynamo/wheelhouse/ /opt/dynamo/wheelhouse/
RUN python3 -m pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl \
    && cd /opt/dynamo/benchmarks \
    && python3 -m pip install --no-cache . \
    && cd - \
    && rm -rf /opt/dynamo/benchmarks

# Install common and test dependencies
RUN --mount=type=bind,source=./container/deps/requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    python3 -m pip install \
        --no-cache \
        --requirement /tmp/requirements.txt \
        --requirement /tmp/requirements.test.txt

## Copy attribution files and launch banner with correct ownership
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

# Copy tests, benchmarks, deploy and components for CI with correct ownership
COPY --chown=dynamo: tests /workspace/tests
COPY --chown=dynamo: examples /workspace/examples
COPY --chown=dynamo: benchmarks /workspace/benchmarks
COPY --chown=dynamo: deploy /workspace/deploy
COPY --chown=dynamo: components/ /workspace/components/
COPY --chown=dynamo: recipes/ /workspace/recipes/

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []

###########################################################
########## Development (run.sh, runs as root user) ########
###########################################################
#
# PURPOSE: Local development environment for use with run.sh (not Dev Container plug-in)
#
# This stage runs as root and provides:
# - Development tools and utilities for local debugging
# - Support for vscode/cursor development outside the Dev Container plug-in
#
# Use this stage if you need a full-featured development environment with extra tools,
# but do not use it with the Dev Container plug-in.

FROM runtime AS dev

ARG WORKSPACE_DIR=/sgl-workspace/dynamo
ARG PYTHON_VERSION

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

# NOTE: SGLang uses system Python (not a virtualenv in framework/runtime stages) to align with
# upstream SGLang Dockerfile: https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile
# For dev stage, we create a lightweight venv with --system-site-packages to satisfy maturin develop
# requirements while still accessing all system-installed packages (sglang, torch, deepep, etc.)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION --system-site-packages

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

USER root
# Install development tools and utilities
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends  \
    # System monitoring and debugging tools
    nvtop \
    htop \
    gdb \
    # Network and system utilities
    wget \
    iproute2 \
    net-tools \
    openssh-client \
    rsync \
    lsof \
    # File and archive utilities
    zip \
    tree \
    # Development and build tools
    vim \
    tmux \
    git \
    git-lfs \
    autoconf \
    automake \
    cmake \
    libtool \
    meson \
    bear \
    ccache \
    less \
    # Language and development support
    clang \
    libclang-dev \
    # Shell and productivity tools
    zsh \
    silversearcher-ag \
    cloc \
    locales \
    # sudo for dev stage
    sudo \
    # NVIDIA tools dependencies
    gnupg && \
    echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update -y && \
    apt-get install -y nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*

# Install clang-format and clangd
RUN curl --retry 3 --retry-delay 2 -LSso /usr/local/bin/clang-format https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format \
    && curl --retry 3 --retry-delay 2 -L https://github.com/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip -o clangd.zip \
    && unzip clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# Editable install of dynamo
COPY pyproject.toml README.md hatch_build.py /workspace/
RUN python3 -m pip install --no-deps -e .

# Install Python development packages
RUN python3 -m pip install --no-cache-dir \
    maturin[patchelf] \
    pytest \
    black \
    isort \
    icdiff \
    scikit_build_core \
    uv \
    pre-commit \
    pandas \
    matplotlib \
    tabulate

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
