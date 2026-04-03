{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/fastvideo_framework.Dockerfile ===
#########################################
######## FastVideo framework ############
#########################################

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS framework

ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG CUTLASS_COMMIT=e67e63c331d6e4b729047c95cf6b92c8454cba89

COPY --from=dynamo_base /bin/uv /bin/uvx /bin/
COPY --from=dynamo_base /usr/local/rustup/ /usr/local/rustup/
COPY --from=dynamo_base /usr/local/cargo/ /usr/local/cargo/

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        g++ \
        ninja-build \
        cmake \
        protobuf-compiler \
        pkg-config \
        clang \
        libclang-dev \
        patchelf \
        git \
        ffmpeg \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

RUN mkdir -p /opt/dynamo/venv && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv venv /opt/dynamo/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/dynamo/venv \
    PATH="/opt/dynamo/venv/bin:/usr/local/cargo/bin:${PATH}"

# FastVideo's fastvideo-kernel sdist does not vendor the CUTLASS submodule
# headers, so install the exact CUTLASS revision pinned by the FastVideo repo.
# Follow-up for parity with the other backends: remove this workaround once the
# upstream package vendors its build headers or publishes wheels that do not
# require patching the image layout.
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git /tmp/cutlass && \
    git -C /tmp/cutlass checkout ${CUTLASS_COMMIT} && \
    cp -a /tmp/cutlass/include/. /usr/local/include/ && \
    rm -rf /tmp/cutlass

RUN --mount=type=bind,source=./container/deps/fastvideo,target=/tmp/deps/fastvideo \
    --mount=type=cache,target=/root/.cache/uv \
    export UV_CACHE_DIR=/root/.cache/uv && \
    cp /tmp/deps/fastvideo/install_fastvideo.sh /tmp/install_fastvideo.sh && \
    chmod +x /tmp/install_fastvideo.sh && \
    CUDA_VERSION=${CUDA_VERSION} \
    WORKSPACE_DIR=/tmp \
    REQUIREMENTS_FILE=/tmp/deps/fastvideo/requirements.fastvideo.txt \
    INSTALL_LOCAL_DYNAMO=false \
    INSTALL_TORCH_FROM_INDEX=false \
    bash /tmp/install_fastvideo.sh
