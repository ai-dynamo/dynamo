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
 && uv pip install pip setuptools packaging ninja psutil uvloop \
 && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Override at build time to target a different GPU architecture, e.g.:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0 9.0a" ...
ARG TORCH_CUDA_ARCH_LIST="10.0 10.0a"

# Lower MAX_JOBS if the build OOMs (machines with <96GB RAM and many CPU cores).
#   docker build --build-arg MAX_JOBS=2 ...
ARG MAX_JOBS=4

# flash-attention ignores TORCH_CUDA_ARCH_LIST and uses its own FLASH_ATTN_CUDA_ARCHS variable.
# Translate from PyTorch format ("10.0 10.0a", space-separated with dots) to flash-attention
# format ("100;100a", semicolon-separated without dots).
RUN export FLASH_ATTN_CUDA_ARCHS=$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's/ /;/g; s/\.//g') \
 && echo "Building flash-attention for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} MAX_JOBS=${MAX_JOBS}" \
 && git clone https://github.com/RandNMR73/flash-attention \
 && cd flash-attention \
 && git switch fa4-compile \
 && uv pip install . --no-build-isolation \
 && uv pip install ./flash_attn/cute \
 && rm -rf ../flash-attention

# Install Dynamo with /v1/videos support.
RUN uv pip install ai-dynamo==1.0.2

# Install FastVideo directly from the public upstream repository.
# Checkout with --recurse-submodules to get the required submodules as well.
RUN echo "Building FastVideo for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install setuptools_scm scikit-build-core cmake ninja \
 && git clone --recurse-submodules https://github.com/hao-ai-lab/FastVideo.git /tmp/FastVideo \
 && uv pip install --no-build-isolation /tmp/FastVideo

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast

WORKDIR /opt/app
COPY . /opt/app/
