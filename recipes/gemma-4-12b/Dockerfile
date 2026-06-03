# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom runtime image for google/gemma-4-12B-it on Dynamo.
#
# google/gemma-4-12B-it is a unified, encoder-free decoder
# (architectures: ["Gemma4UnifiedForConditionalGeneration"], model_type:
# gemma4_unified). Released vLLM does NOT yet support this architecture; it is
# added by the (currently unmerged) vLLM PR #44429 — a pure-Python change
# (one model file + a registry line), no CUDA kernels. We therefore layer:
#
#   1. vLLM built from PR #44429 (pinned SHA), reusing the precompiled kernels
#      from the upstream base image's vLLM build (VLLM_USE_PRECOMPILED=1), so
#      only the Python sources are (re)installed — no kernel compile.
#   2. transformers from a `main` commit that ships transformers.models.gemma4_unified.
#   3. The ai-dynamo Python package (dynamo.frontend, dynamo.vllm, ...) from
#      https://pypi.nvidia.com/ai-dynamo/ with --no-deps, plus its Rust _core
#      extension wheel (ai-dynamo-runtime) and the multimodal runtime extras.
#      (Same approach as recipes/nemotron-3-nano-omni/Dockerfile.)
#
# Build (from the repo root):
#   docker build --platform=linux/arm64 \
#                -t <registry>/gemma4-12b-vllm:latest \
#                -f recipes/gemma-4-12b/Dockerfile \
#                .
#
# Override defaults with --build-arg, e.g.:
#   --build-arg BASE_IMAGE=vllm/vllm-openai:v0.22.0
#   --build-arg DYNAMO_VERSION=1.2.0.post1
#   --build-arg VLLM_PR_SHA=25f921b008768712e99611cc5d942178637359c3
#   --build-arg TRANSFORMERS_REF=d3f05911abb216047a00aba79c8543c73633db05
#
# ALTERNATIVE (offline / reproducible): instead of cloning the PR here, build a
# wheel in ~/vllm (`VLLM_USE_PRECOMPILED=1 uv build --wheel`) and COPY it in:
#   COPY dist/vllm-*.whl /tmp/   &&   pip install --no-deps /tmp/vllm-*.whl
# See README.md "Build the Container".

# Align BASE_IMAGE's vLLM minor with PR #44429's base (it targets vLLM main,
# post-0.22.0). The base supplies CUDA/torch/system libs; VLLM_USE_PRECOMPILED
# pulls the matching precompiled vLLM wheel for the pinned commit.
ARG BASE_IMAGE="vllm/vllm-openai:v0.22.0"
# MUST be >= the release carrying Dynamo's gemma4 tool-call/reasoning parsers
# (added 2026-04-30, PR #8852). 1.2.0.post1 is the matching 0.22-era release;
# older wheels (e.g. 1.2.0.dev20260427) reject `--dyn-tool-call-parser gemma4`.
# Avoid 1.3.0.dev* here unless BASE_IMAGE's vLLM is bumped to match.
ARG DYNAMO_VERSION="1.2.0.post1"
# PR #44429 is fetched via its head ref, then checked out at a pinned commit for
# reproducibility. VLLM_PR_SHA must be an ancestor of (or equal to) the fetched
# ref tip; override both if the PR is force-pushed past this commit.
ARG VLLM_PR_REF="refs/pull/44429/head"
ARG VLLM_PR_SHA="25f921b008768712e99611cc5d942178637359c3"
# transformers `main` commit that ships transformers.models.gemma4_unified.
ARG TRANSFORMERS_REF="d3f05911abb216047a00aba79c8543c73633db05"

FROM ${BASE_IMAGE}
USER root

ARG DYNAMO_VERSION
ARG VLLM_PR_REF
ARG VLLM_PR_SHA
ARG TRANSFORMERS_REF

# Build tooling needed to clone + install vLLM from source. git for the clone;
# the precompiled path skips the CUDA kernel compile.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        jq \
        sox \
        libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# 1. vLLM from PR #44429 — pure-Python arch addition. VLLM_USE_PRECOMPILED=1
#    reuses precompiled kernels (downloads the matching wheel for the commit),
#    so this installs Python sources only — no long ARM kernel build.
#    If no precompiled wheel exists for this commit/arch, drop the env var to
#    fall back to a full source build (slow but correct on the GH200).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    git clone https://github.com/vllm-project/vllm.git /opt/vllm-src; \
    cd /opt/vllm-src; \
    git fetch origin "${VLLM_PR_REF}:gemma4-unified-pr44429"; \
    git checkout "${VLLM_PR_SHA}"; \
    VLLM_USE_PRECOMPILED=1 pip install .; \
    rm -rf /opt/vllm-src

# 2. transformers carrying models/gemma4_unified (config/processor/modeling).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
      "transformers @ git+https://github.com/huggingface/transformers@${TRANSFORMERS_REF}"

# 3a. ai-dynamo Python package with --no-deps so the PR vLLM / torch / cuda
#     stack installed above is preserved (ai-dynamo[vllm] would otherwise try to
#     re-pin vllm), and so we skip nixl/ray that an aggregated single-GPU recipe
#     does not need.
RUN pip install --no-cache-dir --no-deps \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo==${DYNAMO_VERSION}

# 3b. ai-dynamo's core deps + the Rust _core extension wheel (ai-dynamo-runtime)
#     the Python package imports natively. kubernetes/pydantic pinned to
#     ai-dynamo's declared compatibility ranges.
RUN pip install --no-cache-dir \
      --extra-index-url https://pypi.nvidia.com \
      ai-dynamo-runtime==${DYNAMO_VERSION} \
      "kubernetes<33.0.0,>=32.0.1" \
      "pydantic<2.13" "pydantic-settings<2.13.0" \
      msgpack msgspec prometheus-client pyzmq

# 3c. Multimodal runtime extras (image today; librosa/soundfile/av cover the
#     audio/video paths for the deferred follow-up). numpy/pillow are also used
#     by the bundled JSONL benchmark generator.
RUN pip install --no-cache-dir \
      blake3 librosa soundfile uvloop \
      av ftfy nvtx sentencepiece \
      "numpy<3" pillow

# Benchmark dataset generator used by data-gen/generate-datasets-job.yaml.
# Keep the build context at the repo root and copy only this small directory.
COPY benchmarks/multimodal/jsonl /workspace/benchmarks/multimodal/jsonl

# Sanity-check at build time that the architecture and processor resolve.
RUN python3 -c "import transformers.models.gemma4_unified; \
from vllm.model_executor.models.registry import ModelRegistry as R; \
assert 'Gemma4UnifiedForConditionalGeneration' in R.get_supported_archs(), \
'Gemma4UnifiedForConditionalGeneration missing from vLLM registry'; \
print('OK: gemma4_unified arch + transformers module present')"

# The base image ships flashinfer-jit-cache 0.6.11.post2 while PR #44429's vLLM
# pulls flashinfer 0.6.12; vLLM's exact-match version check then aborts model
# inspection ("Gemma4UnifiedForConditionalGeneration failed to be inspected").
# The delta is patch-level and generation is verified correct, so bypass the
# check. (If you prefer a hard pin, align flashinfer-jit-cache to the flashinfer
# version instead and drop this.)
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# vllm/vllm-openai's default ENTRYPOINT runs `vllm serve`; reset it so the
# image behaves as a plain dynamo runtime image.
ENTRYPOINT ["/bin/bash"]
