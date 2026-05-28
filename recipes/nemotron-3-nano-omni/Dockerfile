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

# ai-dynamo Python package (dynamo.frontend, dynamo.vllm, ...) installed with
# --no-deps so that:
#   - the base image's vLLM 0.20 / torch / cuda stack is preserved (otherwise
#     pip would try to re-install vllm[flashinfer,otel,runai]==0.20.0 which is
#     declared by ai-dynamo[vllm]), and
#   - we skip `nixl` (KV transport, only used for disagg) and `ray` (multi-node
#     orchestration), neither of which an aggregated single-GPU recipe needs.
RUN pip install --no-cache-dir --no-deps \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo==${DYNAMO_VERSION}

# ai-dynamo's declared core dependencies, plus the Rust _core extension wheel
# (`ai-dynamo-runtime`) that the Python package imports natively. `kubernetes`,
# `pydantic`, and `pydantic-settings` are explicitly pinned because the latest
# unpinned versions (kubernetes 35.x, pydantic 2.13.x) violate ai-dynamo's
# declared compatibility ranges.
RUN pip install --no-cache-dir \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo-runtime==${DYNAMO_VERSION} \
      "kubernetes<33.0.0,>=32.0.1" \
      "pydantic<2.13" "pydantic-settings<2.13.0" \
      msgpack msgspec prometheus-client pyzmq transformers

# ai-dynamo[vllm] extras minus nixl + ray (handled above), plus the
# multimodal-video Python deps (av/ftfy/nvtx/sentencepiece) that the Nemotron
# Nano Omni model needs at runtime but which aren't pulled in by any extra.
RUN pip install --no-cache-dir \
      blake3 librosa soundfile uvloop \
      av ftfy nvtx sentencepiece

# vllm/vllm-openai's default ENTRYPOINT runs `vllm serve`; reset it so the
# image behaves as a plain dynamo runtime image.
ENTRYPOINT ["/bin/bash"]
