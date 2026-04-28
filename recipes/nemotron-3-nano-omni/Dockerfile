# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Layer the ai-dynamo wheel (and the omni multimodal extras) onto an upstream
# vLLM image. Dynamo is installed from https://pypi.nvidia.com/ai-dynamo/ —
# no source build, no Rust toolchain. Match DYNAMO_VERSION to a nightly that
# targets the same vLLM minor as BASE_IMAGE (see the wheel's METADATA for the
# pinned vllm version).
#
# Build:
#   docker build -t <registry>/nemotron-omni-vllm:latest \
#                -f recipes/nemotron-3-nano-omni/Dockerfile \
#                recipes/nemotron-3-nano-omni
#
# Override defaults with --build-arg, e.g.:
#   --build-arg BASE_IMAGE=vllm/vllm-openai:v0.20.0
#   --build-arg DYNAMO_VERSION=1.2.0.dev20260427

ARG BASE_IMAGE="vllm/vllm-openai:v0.20.0"
ARG DYNAMO_VERSION="1.2.0.dev20260427"

FROM ${BASE_IMAGE}
USER root

ARG DYNAMO_VERSION

# Install ai-dynamo without its declared deps (so the base image's vLLM/torch
# stack is preserved and we skip the heavy `nixl` and `ray` extras we don't
# need for an aggregated single-GPU deployment), then install the runtime
# wheel and the deps we actually want explicitly.
RUN pip install --no-cache-dir --no-deps \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo==${DYNAMO_VERSION} && \
    pip install --no-cache-dir \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo-runtime==${DYNAMO_VERSION} \
      "kubernetes<33.0.0,>=32.0.1" \
      "pydantic<2.13" "pydantic-settings<2.13.0" \
      msgpack msgspec prometheus-client pyzmq transformers \
      blake3 librosa soundfile uvloop \
      av ftfy nvtx sentencepiece

# vllm/vllm-openai's default ENTRYPOINT runs `vllm serve`; reset it so the
# image behaves as a plain dynamo runtime image.
ENTRYPOINT ["/bin/bash"]
