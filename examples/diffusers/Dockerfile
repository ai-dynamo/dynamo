# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared runtime image for Dynamo frontend and FastVideo workers.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools wheel packaging ninja psutil uvloop \
 && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# FA4 (flash-attention from-source build) is deferred for now; the worker runs
# on FastVideo's default attention backends without it.

# Install Dynamo with /v1/videos support.
RUN uv pip install ai-dynamo==1.2.1

# Install FastVideo, which provides the typed fastvideo.api surface
# and NVFP4 transformer quantization support.
RUN uv pip install fastvideo==0.2.0

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast

WORKDIR /opt/app
COPY . /opt/app/
