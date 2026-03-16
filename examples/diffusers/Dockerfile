# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared runtime image for Dynamo frontend and FastVideo workers.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools packaging ninja psutil uvloop \
 && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# flash-attn compilation is memory-intensive. If the build OOMs, lower MAX_JOBS.
# The flash-attn install notes call this out for machines with <96GB RAM and many CPU cores.
RUN git clone https://github.com/RandNMR73/flash-attention \
 && cd flash-attention \
 && git switch fa4-compile \
 && TORCH_CUDA_ARCH_LIST="10.0 10.0a" MAX_JOBS=4 uv pip install . --no-build-isolation \
 && TORCH_CUDA_ARCH_LIST="10.0 10.0a" MAX_JOBS=4 uv pip install ./flash_attn/cute \
 && rm -rf ../flash-attention

# Install Dynamo with /v1/videos support.
RUN uv pip install 'git+https://github.com/ai-dynamo/dynamo@release/1.0.0#subdirectory=lib/bindings/python' \
 && uv pip install 'git+https://github.com/ai-dynamo/dynamo@release/1.0.0'

# Install FastVideo directly from the public upstream repository.
# Checkout with --recurse-submodules to get the required submodules as well.
RUN . /opt/dynamo/venv/bin/activate \
 && uv pip install setuptools_scm scikit-build-core cmake ninja \
 && git clone --recurse-submodules https://github.com/hao-ai-lab/FastVideo.git /tmp/FastVideo \
 && TORCH_CUDA_ARCH_LIST="10.0 10.0a" uv pip install --no-build-isolation /tmp/FastVideo

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast

WORKDIR /opt/app
COPY . /opt/app/
