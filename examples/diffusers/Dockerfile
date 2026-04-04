# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compatibility image for local FastVideo compose flows.
# The canonical CI/release image now comes from container/render.py --framework fastvideo.
# Follow-up for parity with the other backends: drop this compatibility image
# once local compose can consume the rendered runtime image path directly.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev clang build-essential cmake pkg-config patchelf \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

RUN uv venv /opt/dynamo/venv --python 3.12

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace/

RUN CUDA_VERSION=13.1 \
    WORKSPACE_DIR=/workspace \
    INSTALL_LOCAL_DYNAMO=true \
    bash /workspace/container/deps/fastvideo/install_fastvideo.sh

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast
