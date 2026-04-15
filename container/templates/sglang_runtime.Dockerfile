{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{% if device == "xpu" %}
FROM framework AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime
{% endif %}

{% if device != "xpu" %}
# NOTE: Unlike vLLM/TRTLLM, the SGLang upstream runtime image already ships with the full CUDA
# toolkit (nvcc, nvlink, ptxas, etc.), so no selective COPY of CUDA binaries is needed here.

# cleanup unnecessary libs (python3-blinker conflicts with pip-installed blinker from Flask/dash)
RUN apt remove -y python3-apt python3-blinker && \
    pip uninstall -y termplotlib
{% endif %}

ARG TARGETARCH
{% if device != "xpu" %}
# This ARG is still utilized for SGLANG Version extraction
ARG RUNTIME_IMAGE_TAG
{% endif %}
WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

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

# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # required for verification of GPG keys
        gnupg2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy attribution files
COPY --chmod=664 --chown=dynamo:0 ATTRIBUTION* LICENSE /workspace/

{% if context.sglang.enable_media_ffmpeg == "true" %}
# Copy ffmpeg
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/
{% endif %}

# Copy wheels first (separate from benchmarks to avoid unnecessary cache invalidation)
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

# NIXL environment and native libraries
{% if device == "xpu" %}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/x86_64-linux-gnu
{% else %}
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib64
{% endif %}
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

# Copy UCX and NIXL native libraries to system directories
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder $NIXL_PREFIX $NIXL_PREFIX
{% if device == "xpu" %}
{# XPU NIXL uses lib/x86_64-linux-gnu; copy to NIXL_LIB_DIR to ensure lib dir is populated #}
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/
{% endif %}

ENV PATH=/usr/local/ucx/bin:$PATH

ENV LD_LIBRARY_PATH=\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
$LD_LIBRARY_PATH

{% if device == "xpu" %}
# For XPU builds SGLANG_VERSION is taken from the pinned SGLANG_REF build argument
ARG SGLANG_REF
ENV SGLANG_VERSION="${SGLANG_REF}"
{% else %}
ENV SGLANG_VERSION="${RUNTIME_IMAGE_TAG%%-*}"
{% endif %}

{% if target not in ("dev", "local-dev") %}
{% if device == "xpu" %}
# XPU: install dynamo + NIXL into the venv inherited from the framework stage
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% else %}
# CUDA: install packages as root to system location (/usr/local/lib/python3.12/dist-packages)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
        sglang==${SGLANG_VERSION}
{% endif %}
{% else %}
# Dev/local-dev: skip dynamo wheel install (users build from source via cargo build + maturin develop).
# Install NIXL wheel only (pre-built C++ binary, not buildable from source).
{% if device == "xpu" %}
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% else %}
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
        sglang==${SGLANG_VERSION}
{% endif %}
{% endif %}

{% if device != "xpu" %}
# Install gpu_memory_service wheel if enabled (CUDA only)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then pip install --no-cache-dir --break-system-packages "$GMS_WHEEL"; fi; \
    fi
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Copy benchmarks after wheel install so benchmarks changes don't invalidate the layer above
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/
{% endif %}

# Install runtime dependencies (common + benchmarks).
# Test and dev dependencies are NOT installed here — they go in the test and dev images.
{% if device == "xpu" %}
RUN --mount=type=bind,source=container/deps/requirements.common.txt,target=/tmp/deps/requirements.common.txt \
    --mount=type=bind,source=container/deps/requirements.benchmark.txt,target=/tmp/deps/requirements.benchmark.txt \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --requirement /tmp/deps/requirements.common.txt \
        --requirement /tmp/deps/requirements.benchmark.txt
{% else %}
RUN --mount=type=bind,source=container/deps/requirements.common.txt,target=/tmp/deps/requirements.common.txt \
    --mount=type=bind,source=container/deps/requirements.benchmark.txt,target=/tmp/deps/requirements.benchmark.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages \
        --requirement /tmp/deps/requirements.common.txt \
        --requirement /tmp/deps/requirements.benchmark.txt \
        sglang==${SGLANG_VERSION} && \
    #TODO: Temporary change until upstream sglang runtime image is updated
    pip install --break-system-packages "urllib3>=2.6.3"
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Install benchmarks and fix permissions (dev/local-dev install from bind-mounted source if needed)
{% if device == "xpu" %}
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    cd /workspace/benchmarks && \
    uv pip install . && \
    chmod -R g+w /workspace/benchmarks
{% else %}
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    cd /workspace/benchmarks && \
    pip install --break-system-packages . && \
    chmod -R g+w /workspace/benchmarks
{% endif %}
{% endif %}

{% if device != "xpu" %}
# Force-reinstall NVIDIA packages in a separate layer so requirements changes don't trigger re-download
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    CUDA_MAJOR=$(nvcc --version | egrep -o 'cuda_[0-9]+' | cut -d_ -f2) && \
    if [ "$CUDA_MAJOR" = "12" ]; then \
        # Install NVIDIA packages that are needed for DeepEP to work properly
        # This is done in the upstream runtime image too, but these packages are overridden in earlier commands
        pip install --break-system-packages --force-reinstall --no-deps \
            nvidia-nccl-cu12==2.28.3 \
            nvidia-cudnn-cu12==9.16.0.29 \
            nvidia-cutlass-dsl==4.3.5; \
    elif [ "$CUDA_MAJOR" = "13" ]; then \
        # CUDA 13: Install CuDNN for PyTorch 2.9.1 compatibility
        pip install --break-system-packages --force-reinstall --no-deps \
            nvidia-nccl-cu13==2.28.3 \
            nvidia-cublas==13.1.0.3 \
            nvidia-cutlass-dsl==4.3.1 \
            nvidia-cudnn-cu13==9.16.0.29; \
    fi
{% endif %}

# Switch back to dynamo user after package installations
USER dynamo

# Copy tests, deploy, and the sglang/common/mocker component subtrees for CI.
# Pattern: COPY --chmod=775 <path>; chmod g+w <path> done later as root because COPY --chmod only affects <path>/*, not <path>
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/sglang /workspace/components/src/dynamo/sglang
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

# Our scripting assumes /workspace is where dynamo is located
# In order to maintain the ability to have sglang and dynamo
# in the same workspace, symlink /workspace to /sgl-workspace/dynamo
USER root

# Fix directory permissions: COPY --chmod only affects contents, not the directory itself
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc && \
{% if device == "xpu" %}
    echo 'source /opt/intel/oneapi/setvars.sh --force' >> /etc/bash.bashrc && \
    echo 'export LD_LIBRARY_PATH=/opt/dynamo/venv/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
{% endif %}
    mkdir -p /sgl-workspace && \
    ln -sf /workspace /sgl-workspace/dynamo

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

{% if device == "xpu" %}
CMD ["bash", "-c", "source /etc/bash.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
