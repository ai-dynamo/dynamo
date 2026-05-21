{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/frontend.Dockerfile ===
##############################################
########## Frontend entrypoint image #########
##############################################
FROM ${EPP_IMAGE} AS epp

# Build `crick` as a wheel in an isolated stage so the C toolchain never
# reaches the final frontend image. aiperf 0.8.0 depends on crick==0.0.8,
# which publishes no manylinux aarch64 wheel — without this, arm64 builds
# fall back to sdist and fail in the final stage where gcc is intentionally
# absent. amd64 has a prebuilt manylinux_x86_64 wheel on PyPI, so the build
# is gated on TARGETARCH=arm64; amd64 ships an empty /wheels and uv pulls
# crick straight from PyPI in the final stage. /wheels is created either
# way so the COPY --from=crick_builder in the final stage always succeeds.
FROM ${FRONTEND_IMAGE} AS crick_builder
ARG PYTHON_VERSION
ARG TARGETARCH
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    mkdir -p /wheels \
    && if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get update -y \
        && apt-get install -y --no-install-recommends \
            ca-certificates \
            gcc \
            libc6-dev \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; \
    fi
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        python${PYTHON_VERSION} -m venv /tmp/buildenv \
        && /tmp/buildenv/bin/pip install --no-cache-dir --upgrade pip wheel \
        && /tmp/buildenv/bin/pip wheel --no-cache-dir --no-deps crick==0.0.8 -w /wheels; \
    fi

FROM ${FRONTEND_IMAGE} AS frontend

ARG PYTHON_VERSION
# Cache apt downloads; sharing=locked avoids apt/dpkg races with concurrent builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # required for EPP
        ca-certificates \
        libstdc++6 \
        # required for verification of GPG keys
        gnupg2 \
        # required for installing dependencies from git repositories
        git \
        git-lfs \
        # Python runtime - required for virtual environment to work
        python${PYTHON_VERSION}-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo /workspace \
    && chown -R dynamo: /opt/dynamo /home/dynamo/.cache /workspace \
    && chmod -R g+w /opt/dynamo /home/dynamo/.cache /workspace

# Set HOME so ModelExpress can find the cache directory
ENV HOME=/home/dynamo
# Switch to dynamo user
USER dynamo
ENV DYNAMO_HOME=/opt/dynamo

WORKDIR /
COPY --chown=dynamo: --from=epp /epp /epp

COPY --chown=dynamo: container/launch_message/frontend.txt /opt/dynamo/.launch_screen
# Copy tests, benchmarks, deploy and components with correct ownership
COPY --chown=dynamo: tests /workspace/tests
COPY --chown=dynamo: examples /workspace/examples
COPY --chown=dynamo: benchmarks /workspace/benchmarks
COPY --chown=dynamo: deploy /workspace/deploy
COPY --chown=dynamo: components/ /workspace/components/
COPY --chown=dynamo: recipes/ /workspace/recipes/
# Copy attribution files with correct ownership
COPY --chown=dynamo: ATTRIBUTION* LICENSE /workspace/

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="/opt/dynamo/venv/bin:$PATH"

# Copy uv from base stage and wheels from wheel_builder (no runtime stage dependency)
COPY --chown=dynamo: --from=dynamo_base /bin/uv /bin/uvx /bin/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/
COPY --chown=dynamo: --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo: --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/
# crick wheel pre-built in the crick_builder stage; see comment near the top.
COPY --chown=dynamo: --from=crick_builder /wheels/ /opt/dynamo/wheelhouse/extra/

# Create virtual environment
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv && \
    mkdir -p /opt/dynamo/venv && \
    uv venv /opt/dynamo/venv --python $PYTHON_VERSION

# Install runtime dependencies (common + frontend).
# Frontend needs tritonclient and its grpcio/protobuf constraints for gRPC serving.
# Test and dev dependencies are NOT installed here — they go in the test and dev images.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=bind,source=./container/deps/requirements.frontend.txt,target=/tmp/requirements.frontend.txt \
    --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install \
        --requirement /tmp/requirements.common.txt \
        --requirement /tmp/requirements.frontend.txt

ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
# In an ideal world, we'd use a mirror of PyPI for much more reliable downloads.
# UV_FIND_LINKS points at the crick wheel pre-built in the crick_builder stage;
# uv prefers it over the sdist on arm64 where no manylinux aarch64 wheel exists.
RUN --mount=type=cache,target=/home/dynamo/.cache/uv,uid=1000,gid=0,mode=0775,sharing=shared \
    export UV_CACHE_DIR=/home/dynamo/.cache/uv UV_FIND_LINKS=/opt/dynamo/wheelhouse/extra && \
    uv pip install \
    /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
    /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
    /opt/dynamo/wheelhouse/nixl/nixl*.whl && \
    if [ "$ENABLE_GPU_MEMORY_SERVICE" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -z "$GMS_WHEEL" ]; then \
            echo "ERROR: ENABLE_GPU_MEMORY_SERVICE is true but no gpu_memory_service wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$GMS_WHEEL"; \
    fi && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -z "$KVBM_WHEEL" ]; then \
            echo "ERROR: ENABLE_KVBM is true but no KVBM wheel found in wheelhouse" >&2; \
            exit 1; \
        fi; \
        uv pip install "$KVBM_WHEEL"; \
    fi && \
    cd /workspace/benchmarks && \
    export UV_GIT_LFS=1 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv pip install .

# Setup environment for all users
USER root
RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'source /opt/dynamo/venv/bin/activate' >> /etc/bash.bashrc && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ENTRYPOINT ["/epp"]
CMD ["/bin/bash"]
