{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dev.Dockerfile ===
# ======================================================================
# STAGE: dynamo_tools for developers
# ======================================================================
# Why this is a separate stage (not merged into `dev`):
# - `dev` is built FROM the framework `runtime` image. Installing lots of tooling with apt in that stage is slow and
#   makes rebuilds expensive when iterating on later dev layers.
# - Keeping tooling installation in `dynamo_tools` lets Docker cache the tools layer independently; `dev` can then
#   pull those binaries/configs in via COPY.
FROM runtime AS dynamo_tools

ARG TARGETARCH
ARG DEVICE

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:${PATH}

USER root
SHELL ["/bin/bash", "-c"]

# NOTE: We intentionally disable the NVIDIA CUDA apt repo for this stage.
# The upstream runtime images may ship CUDA apt sources that occasionally go out of sync (mirror updates),
# causing apt-get update to fail with "File has unexpected size ... Mirror sync in progress".
# This stage only installs generic developer tools that are available from Ubuntu repos, so CUDA repos are unnecessary.
#
# We also add a small retry/backoff to make transient apt metadata issues less disruptive.
# Estimated layer size: ~800MB–1.0GB (build-essential+clang ~500MB, the rest ~300MB)
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eux; \
    if [ -d /etc/apt/sources.list.d ]; then \
        mkdir -p /tmp/apt-disabled; \
        for f in /etc/apt/sources.list.d/*.list; do \
            [ -e "$f" ] || continue; \
            if grep -q "developer.download.nvidia.com/compute/cuda/repos" "$f"; then \
                mv "$f" "/tmp/apt-disabled/$(basename "$f")"; \
            fi; \
        done; \
    fi; \
    for i in 1 2 3 4 5; do \
        apt-get update -y && break; \
        rm -rf /var/lib/apt/lists/*; \
        sleep $((i * 5)); \
    done; \
    apt-get install -y --no-install-recommends \
        # Core CLI utilities
        ca-certificates \
        curl \
        wget \
        git \
        git-lfs \
        less \
        grep \
        sed \
        # Editors / shells
        vim \
        nano \
        htop \
        tmux \
        screen \
        zsh \
        fish \
        bash-completion \
        # Networking / transfers
        net-tools \
        openssh-client \
        iproute2 \
        iputils-ping \
        zip \
        unzip \
        rsync \
        # Build toolchain
        build-essential \
        cmake \
        autoconf \
        automake \
        libtool \
        meson \
        ninja-build \
        pybind11-dev \
        pkg-config \
        protobuf-compiler \
        # Debugging / tracing
        gdb \
        valgrind \
        strace \
        ltrace \
        # JSON/YAML + filesystem helpers
        jq \
        yq \
        tree \
        fd-find \
        ripgrep \
        # Privilege escalation + crypto tooling
        sudo \
        gnupg2 \
        gnupg1 \
        # GPU / perf helpers
        nvtop \
        # Python
        python3 \
        python3-pip \
        python3-venv \
        # Native deps for Python/Rust wheels
        patchelf \
        clang \
        libclang-dev \
        libfontconfig-dev && \
    rm -rf /var/lib/apt/lists/* && \
    # Initialize Git LFS for the dynamo user (required for requirements with lfs=true)
    git lfs install

# Install awk separately with fault tolerance (~2MB).
# awk is a virtual package with multiple implementations (gawk, mawk, original-awk).
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    (apt-get update && \
     (apt-get install -y --no-install-recommends gawk || \
      apt-get install -y --no-install-recommends mawk || \
      apt-get install -y --no-install-recommends original-awk || \
      echo "Warning: Could not install any awk implementation") && \
     rm -rf /var/lib/apt/lists/*) && \
    (command -v awk >/dev/null 2>&1 && echo "awk available: $(command -v awk)" || echo "awk not available")

# Add NVIDIA devtools repository and install development tools (nsight-systems).
# Estimated layer size: ~500MB–1.5GB (nsight-systems is a full profiling suite)
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    wget -qO - "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/${TARGETARCH}/nvidia.pub" \
        | gpg --dearmor -o /etc/apt/keyrings/nvidia-devtools.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2404/${TARGETARCH} /" \
        | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-2025.5.1 && \
    rm -rf /var/lib/apt/lists/*

# TODO: Add GitHub CLI (gh) for development. Estimated layer size: ~50MB

# ======================================================================
# TARGET: dev (root-based development)
# ======================================================================
#
# USAGE: This dev image ships /workspace EMPTY. You MUST:
#
#   1) Bind-mount your Dynamo repo checkout into the container:
#        docker run --gpus all -v /path/to/dynamo:/workspace ...
#
#   2) Build from source inside the container:
#        cargo build --features dynamo-llm/block-manager
#        cd /workspace/lib/bindings/python && maturin develop --uv
#        uv pip install --no-deps -e /workspace
#
# The pre-built ai-dynamo / ai-dynamo-runtime wheels from the runtime
# stage are uninstalled below to avoid conflicts with the source build.
# ======================================================================
FROM runtime AS dev

# Redeclare ARGs for use in this stage
ARG FRAMEWORK

USER root

# Redeclare build args for use in this stage
ARG PYTHON_VERSION

# Ensure the runtime stage always has /usr/bin/python3.
# - vLLM/TRTLLM runtime images may only have Python in /opt/dynamo/venv/bin/{python,python3}
# - SGLang runtime images typically have /usr/bin/python3 already
# - framework=none runtime stage now installs /usr/bin/python3
RUN if [ ! -e /usr/bin/python3 ]; then \
        if [ -x /opt/dynamo/venv/bin/python3 ]; then \
            ln -s /opt/dynamo/venv/bin/python3 /usr/bin/python3; \
        elif [ -x /opt/dynamo/venv/bin/python ]; then \
            ln -s /opt/dynamo/venv/bin/python /usr/bin/python3; \
        elif command -v python3 >/dev/null 2>&1; then \
            ln -s $(command -v python3) /usr/bin/python3; \
        elif command -v python >/dev/null 2>&1; then \
            ln -s $(command -v python) /usr/bin/python3; \
        else \
            echo "ERROR: Could not find Python to symlink to /usr/bin/python3" >&2; \
            exit 1; \
        fi; \
    fi

# Copy UCX and NIXL libraries for dev stage compilation.
# The upstream SGLang runtime image doesn't include NIXL, but cargo build needs to link against
# -lnixl, -lnixl_build, and -lnixl_common. Runtime stage doesn't need this since it uses pre-built
# wheels, but dev stage needs it for maturin develop and cargo build from source.
# - SGLang: Copy NIXL/UCX/libfabric/gdrcopy binaries from wheel_builder (not in upstream lmsysorg/sglang runtime).
# - vllm/trtllm/none: NIXL/UCX are already present in runtime (no-op).
ARG TARGETARCH
RUN --mount=from=wheel_builder,target=/wheel_builder \
    if [ "${FRAMEWORK}" = "sglang" ]; then \
        if [ -d /wheel_builder/usr/local/ucx ] && [ -d /wheel_builder/opt/nvidia/nvda_nixl ]; then \
            mkdir -p /opt/nvidia /usr/include /usr/lib64 /etc/ld.so.conf.d; \
            cp -r /wheel_builder/opt/nvidia/nvda_nixl /opt/nvidia/; \
            cp -r /wheel_builder/usr/local/ucx /usr/local/; \
            cp -r /wheel_builder/usr/local/libfabric /usr/local/; \
            cp /wheel_builder/usr/include/gdrapi.h /usr/include/; \
            cp /wheel_builder/usr/lib64/libgdrapi.so* /usr/lib64/; \
            echo "/usr/lib64" >> /etc/ld.so.conf.d/gdrcopy.conf; \
        fi; \
    fi

{% if device == "xpu" %}
ENV NIXL_LIB_DIR=/opt/intel/intel_nixl/lib/x86_64-linux-gnu  \
    NIXL_PLUGIN_DIR=/opt/intel/intel_nixl/lib/x86_64-linux-gnu/plugins \
    NIXL_PREFIX=/opt/intel/intel_nixl
{% else %}
# NIXL is installed under lib64 (manylinux/AlmaLinux convention used by the wheel_builder).
# All frameworks reference NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64.
# For vllm/trtllm/none: This resets the same values already set in runtime (no harm).
# For sglang: This sets them for the first time (required).
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins

# Set universal CUDA development environment variables (all frameworks)
# vLLM: Dockerfile.vllm line 533, 597
# TRT-LLM: Dockerfile.trtllm lines 600-606
ENV CUDA_HOME=/usr/local/cuda \
    CPATH=/usr/local/cuda/include \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TRITON_CUPTI_PATH=/usr/local/cuda/include \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUDART_PATH=/usr/local/cuda/include \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% endif %}

# Base LD_LIBRARY_PATH with universal paths (all frameworks have these)
# Framework-specific paths are conditionally added in /etc/profile.d/50-framework-paths.sh
ARG PYTHON_VERSION
ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${LD_LIBRARY_PATH}

# Copy shell profile script for framework-specific environment variables
# This script conditionally adds PATH/LD_LIBRARY_PATH entries based on what exists
COPY --chmod=755 container/dev/50-framework-paths.sh /etc/profile.d/50-framework-paths.sh

# Set umask for group-writable files in dev stage (runs as root)
RUN mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]

# Developer tools are installed in the dynamo_tools layer and copied into the runtime-based dev image.
# This keeps dev builds fast and avoids apt-get in runtime-derived stages.
#
# IMPORTANT: Do not clobber runtime /usr/bin/python3 (SGLang depends on system python3 being present).
# We stash the pre-tools python3 (which may be a real binary or a symlink we created earlier for vLLM/TRTLLM)
# and restore it after copying toolchains from dynamo_tools.
RUN if [ -e /usr/bin/python3 ]; then cp -a /usr/bin/python3 /tmp/python3.pretools; fi
COPY --from=dynamo_tools /usr/bin/ /usr/bin/
COPY --from=dynamo_tools /usr/sbin/ /usr/sbin/
COPY --from=dynamo_tools /usr/lib/ /usr/lib/
COPY --from=dynamo_tools /usr/libexec/ /usr/libexec/
COPY --from=dynamo_tools /lib/ /lib/
COPY --from=dynamo_tools /usr/share/ /usr/share/
COPY --from=dynamo_tools /etc/alternatives/ /etc/alternatives/
COPY --from=dynamo_tools /etc/bash_completion.d/ /etc/bash_completion.d/
COPY --from=dynamo_tools /etc/sudoers /etc/sudoers
COPY --from=dynamo_tools /etc/sudoers.d/ /etc/sudoers.d/
COPY --from=dynamo_tools /opt/nvidia/ /opt/nvidia/

# Restore the pre-tools python3 (keeps SGLang system python intact and avoids venv symlink loops).
RUN if [ -e /tmp/python3.pretools ]; then cp -af /tmp/python3.pretools /usr/bin/python3; fi

ARG WORKSPACE_DIR=/workspace

# Dev environment variables (aligned with framework dev stages)
# Framework-specific PATH additions are handled in /etc/profile.d/50-framework-paths.sh
ENV WORKSPACE_DIR=${WORKSPACE_DIR} \
    DYNAMO_HOME=${WORKSPACE_DIR} \
    RUSTUP_HOME=/home/dynamo/.rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/workspace/target \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/cargo/bin:$PATH

# Copy Rust/Cargo/Maturin from the concatenated framework stages.
# - Rust/Cargo: from `wheel_builder` (already installed there)
# - maturin: from `wheel_builder` venv (installed there via uv pip)
COPY --from=wheel_builder --chown=dynamo:0 --chmod=775 /usr/local/rustup /home/dynamo/.rustup
COPY --from=wheel_builder --chown=dynamo:0 --chmod=775 /usr/local/cargo /usr/local/cargo
COPY --from=wheel_builder --chown=dynamo:0 --chmod=775 /workspace/.venv/bin/maturin /usr/local/bin/maturin

{% if framework == "sglang" %}
# SGLang: Create venv with --system-site-packages to inherit runtime packages
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /tmp/uv-binary
RUN mkdir -p /opt/dynamo/venv && \
    python3 -m venv --system-site-packages /opt/dynamo/venv && \
    cp -r --no-preserve=mode /usr/local/lib/python${PYTHON_VERSION}/dist-packages/* \
          /opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/ && \
    cp /tmp/uv-binary /opt/dynamo/venv/bin/uv && \
    chmod +x /opt/dynamo/venv/bin/uv && \
    pip install --ignore-installed maturin[patchelf]
{% elif framework == "dynamo" %}
# framework=none: Create venv if runtime stage didn't already provide one
RUN if [ ! -d /opt/dynamo/venv ]; then \
        mkdir -p /opt/dynamo && \
        python3 -m venv /opt/dynamo/venv; \
    fi
{% endif %}

# Initialize Git LFS for the dynamo user (required for requirements with lfs=true)
RUN git lfs install

# Install only the ADDITIONAL dev/test dependencies.
# Runtime deps (common, framework, planner, benchmark) are already installed
# in the parent runtime image — re-resolving them here would risk version drift.
# SGLang specific: Reinstall pytest to ensure venv has pytest executable with correct shebang
ARG FRAMEWORK
RUN --mount=type=bind,source=./container/deps/requirements.dev.txt,target=/tmp/requirements.dev.txt \
    --mount=type=bind,source=./container/deps/requirements.test.txt,target=/tmp/requirements.test.txt \
    # Cache uv downloads; uv handles its own locking for this cache.
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.dev.txt \
        --requirement /tmp/requirements.test.txt && \
    if [ "${FRAMEWORK}" = "sglang" ]; then \
        uv pip install --force-reinstall --no-deps pytest; \
    fi

# Copy entire workspace (old design - simpler for CI)
# .dockerignore filters out unwanted files (.git, build artifacts, etc.)
WORKDIR ${WORKSPACE_DIR}
# We don't actually need /workspace because for development, this must be mounted as a volume.
#COPY --chmod=775 --chown=dynamo:0 ./ ${WORKSPACE_DIR}/

RUN mkdir -p ${WORKSPACE_DIR} && chmod g+w ${WORKSPACE_DIR}

# Remove pre-built dynamo packages inherited from the runtime stage.
# The dev image builds from source, so these would conflict with the editable installs.
# NOTE: This does NOT reclaim disk space in the image (files still exist in lower layers).
# Space is only recovered if the image is later squashed / compacted (e.g. docker-squash,
# `docker build --squash`, or export/import).
RUN uv pip uninstall ai-dynamo ai-dynamo-runtime kvbm 2>/dev/null || true

# Install maturin only (no editable install of the dynamo package).
# /workspace is empty at build time — the repo is bind-mounted at container start, not COPYed.
# `uv pip install -e .` would fail here because there is no pyproject.toml in /workspace yet.
# The editable install must be done at runtime after the volume mount (e.g. `maturin develop`).
RUN if command -v uv >/dev/null 2>&1; then \
        uv pip install maturin[patchelf] ; \
    else \
        python3 -m pip install maturin[patchelf] ; \
    fi

# Set commit SHA for tests (passed via docker build as --build-arg)
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

# Setup dev launch banner (displayed on interactive shell entry)
RUN --mount=type=bind,source=./container/launch_message/dev.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

{% if device == "xpu" %}
SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
