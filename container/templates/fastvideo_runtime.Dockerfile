{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/fastvideo_runtime.Dockerfile ===
#######################################
######## FastVideo runtime ############
#######################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS runtime

ARG PYTHON_VERSION
WORKDIR /workspace

# Copy runtime utilities from shared stages.
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /bin/uv /bin/uvx /bin/
# Keep the runtime image CUDA footprint controlled: copy only the development
# tools and libraries FastVideo needs at runtime (instead of the full tree).
COPY --from=framework /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc
COPY --from=framework /usr/local/cuda/bin/nvlink /usr/local/cuda/bin/nvlink
COPY --from=framework /usr/local/cuda/bin/cudafe++ /usr/local/cuda/bin/cudafe++
COPY --from=framework /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
COPY --from=framework /usr/local/cuda/bin/fatbinary /usr/local/cuda/bin/fatbinary
COPY --from=framework /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/cuobjdump
COPY --from=framework /usr/local/cuda/bin/nvdisasm /usr/local/cuda/bin/nvdisasm
COPY --from=framework /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=framework /usr/local/cuda/nvvm /usr/local/cuda/nvvm
COPY --from=framework /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=framework /usr/local/cuda/lib64/libcupti* /usr/local/cuda/lib64/
COPY --from=framework /usr/local/lib/lib* /usr/local/lib/

ENV PATH="/usr/local/bin/etcd/:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda \
    TRITON_CUPTI_PATH=/usr/local/cuda/include \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUDART_PATH=/usr/local/cuda/include

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo /workspace \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

# NIXL environment variables
ENV NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    CARGO_TARGET_DIR=/opt/dynamo/target

ENV LD_LIBRARY_PATH=\
${NIXL_LIB_DIR}:\
${NIXL_PLUGIN_DIR}:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${LD_LIBRARY_PATH}

# Copy UCX, NIXL, and built Dynamo wheels
COPY --chown=dynamo: --from=wheel_builder /usr/local/ucx/ /usr/local/ucx/
COPY --chown=dynamo: --from=wheel_builder ${NIXL_PREFIX}/ ${NIXL_PREFIX}/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    rm -rf /var/cache/apt/archives/partial/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        g++ \
        ninja-build \
        git \
        git-lfs \
        ca-certificates \
        libnuma1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

USER dynamo
ENV HOME=/home/dynamo

SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]
ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}" \
    LD_LIBRARY_PATH="${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:/opt/dynamo/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${LD_LIBRARY_PATH}" \
    FASTVIDEO_VIDEO_CODEC=libx264 \
    FASTVIDEO_X264_PRESET=ultrafast

COPY --chmod=775 --chown=dynamo:0 --from=framework ${VIRTUAL_ENV} ${VIRTUAL_ENV}

{% if target not in ("dev", "local-dev") %}
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% else %}
# Dev/local-dev: skip Dynamo wheel install because those images build Dynamo
# from source after the workspace is mounted. Install the pre-built NIXL wheel
# only since it is still needed by the development environment.
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install /opt/dynamo/wheelhouse/nixl/nixl*.whl
{% endif %}

# Install shared runtime dependencies (common + planner + benchmark).
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.benchmark.txt,target=/tmp/requirements.benchmark.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --index-strategy unsafe-best-match \
        --requirement /tmp/requirements.common.txt \
        --requirement /tmp/requirements.planner.txt \
        --requirement /tmp/requirements.benchmark.txt

{% if target not in ("dev", "local-dev") %}
# Copy benchmarks after dependency install so benchmark source changes do not
# invalidate the shared runtime dependency layer above.
COPY --chmod=775 --chown=dynamo:0 benchmarks/ /workspace/benchmarks/

RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775 \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    cd /workspace/benchmarks && \
    uv pip install . && \
    chmod -R g+w /workspace/benchmarks
{% endif %}

ARG WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}

# Copy tests, deploy and components for CI with correct ownership
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/fastvideo /workspace/components/src/dynamo/fastvideo
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
RUN chmod g+w ${WORKSPACE_DIR}

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
