{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/dynamo_runtime.Dockerfile ===
#######################################
########## Runtime image ##############
#######################################

FROM dynamo_base AS runtime

ARG PYTHON_VERSION

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

ENV CARGO_TARGET_DIR=/opt/dynamo/target

# Always copy FFmpeg so libs are available for Rust checks in CI.
# libvpx.so* is included because the in-tree ffmpeg is built with --enable-libvpx,
# so libavcodec.so has a runtime dependency on libvpx.so.9.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so /tmp/usr/local/lib/libsw*.so /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/ && \
    ldconfig

# Runtime test images rebuild Rust crates, so carry the same version-matched
# protoc binary and well-known schemas used by the wheel builder.
COPY --from=wheel_builder /usr/local/bin/protoc /usr/local/bin/protoc
COPY --from=wheel_builder /usr/local/include/google/protobuf/ /usr/local/include/google/protobuf/
ENV PROTOC=/usr/local/bin/protoc
RUN protoc --version && test -f /usr/local/include/google/protobuf/struct.proto

COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

# Install Python for framework=none runtime (cuda-dl-base doesn't include Python)
# This is needed to create venv and install dynamo packages
ARG PYTHON_VERSION
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
# Clear partial downloads first to avoid stale rename failures from prior interrupted builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    rm -rf /var/cache/apt/archives/partial/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cmake \
        pkg-config \
        clang \
        libclang-dev \
        patchelf \
        git \
        git-lfs \
        libjemalloc2 \
        libturbojpeg && \
    ldconfig && \
    ldconfig -p | grep -q 'libturbojpeg.so.0' && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# libjemalloc2 is installed above so user may opt into jemalloc for memory allocation
# tuning. We deliberately do NOT set ENV LD_PRELOAD here — that would force jemalloc
# on every process in the image.

# Switch to dynamo user and create virtual environment
USER dynamo
ENV HOME=/home/dynamo

# Create and activate virtual environment
# Use login shell to pick up umask 002 from /etc/profile.d/00-umask.sh for group-writable files
SHELL ["/bin/bash", "-l", "-o", "pipefail", "-c"]
# Cache uv downloads; uv handles its own locking for the cache.
RUN --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:${PATH}"

{% if target not in ("dev", "local-dev") %}
# Install dynamo wheels (runtime packages only, no test dependencies)
# uv handles its own locking for the cache, no need to add sharing=locked
RUN --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl
{% else %}
# Dev/local-dev: skip dynamo wheel install (users build from source via cargo build + maturin develop).
{% endif %}

# NIXL from PyPI (all targets). The meta package requires both CUDA backends,
# so install the backend first and the meta package with --no-deps.
ARG NIXL_REF
ARG CUDA_MAJOR
RUN --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    echo "${NIXL_REF}" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+$' || { echo "NIXL_REF must be a vX.Y.Z release tag; got '${NIXL_REF}'" >&2; exit 1; } && \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    uv pip install "nixl-cu${CUDA_MAJOR}==${NIXL_REF#v}" && \
    uv pip install --no-deps "nixl==${NIXL_REF#v}" && \
    uv pip show nixl "nixl-cu${CUDA_MAJOR}" | grep -E '^(Name|Version)' | tee /opt/dynamo/nixl-versions.txt

# Register the NIXL wheel's libraries with the loader so the Rust bindings'
# dlopen("libnixl_capi.so") resolves. NIXL finds its plugins beside them.
USER root
RUN --mount=type=bind,source=./container/deps/register_nixl_libs.py,target=/tmp/register_nixl_libs.py,readonly \
    /opt/dynamo/venv/bin/python /tmp/register_nixl_libs.py --cuda-major "${CUDA_MAJOR}"
USER dynamo

# Install gpu_memory_service wheel if enabled (all targets)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install "$GMS_WHEEL"; fi; \
    fi

# Initialize Git LFS (required for git+https dependencies with LFS artifacts)
RUN git lfs install

# Install runtime dependencies (common + planner + frontend).
# Frontend deps (tritonclient + grpcio/protobuf pins) are installed here so the resolver
# sees all constraints in one pass, avoiding grpcio downgrades in the test layer.
# Test and dev dependencies are NOT installed here — they go in the test and dev images.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.planner.txt,target=/tmp/requirements.planner.txt \
    --mount=type=bind,source=./container/deps/requirements.frontend.txt,target=/tmp/requirements.frontend.txt \
    --mount=type=cache,id=uv-dynamo-{{ context.dynamo.uv_version }},target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.common.txt \
        --requirement /tmp/requirements.planner.txt \
        --requirement /tmp/requirements.frontend.txt

# TODO: skip /workspace COPY for dev/local-dev (bind-mounted from host, gets shadowed)
# Copy workspace source code
ARG WORKSPACE_DIR=/workspace
WORKDIR ${WORKSPACE_DIR}
COPY --chmod=775 --chown=dynamo:0 ./ ${WORKSPACE_DIR}/

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []


# Compliance stages are intentionally NOT included here: dynamo-runtime is an
# UNPUBLISHED wheel-builder image (release.yml ships vllm/sglang/trtllm-runtime,
# frontend, operator, planner, snapshot-agent — not this), so it carries no
# shipped-NOTICES obligation. The Rust-crate attribution that matters lives in
# the published wheels themselves (maturin SBOM + bundled THIRD-PARTY-RUST-
# LICENSES, see wheel_builder) and in the published framework images' /legal.
