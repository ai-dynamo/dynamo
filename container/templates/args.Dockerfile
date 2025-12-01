##################################
########## Build Arguments ########
##################################

# Base image configuration
ARG BASE_IMAGE={{ context.dynamo_base.base_image }}
# TODO OPS-612: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
ARG BASE_IMAGE_TAG={{ context.dynamo_base.base_image_tag }}
{% if framework == "vllm" -%}
ARG RUNTIME_IMAGE={{ context[framework][platform].runtime_image }}
ARG RUNTIME_IMAGE_TAG={{ context[framework][platform].runtime_image_tag }}
{% elif framework != "none" -%}
ARG RUNTIME_IMAGE={{ context[framework].runtime_image }}
ARG RUNTIME_IMAGE_TAG={{ context[framework].runtime_image_tag }}
{%- endif %}
# Build configuration
ARG ENABLE_KVBM={{ context.dynamo_base.enable_kvbm }}
ARG CARGO_BUILD_JOBS

# Define general architecture ARGs for supporting both x86 and aarch64 builds.
#   ARCH: Used for package suffixes (e.g., amd64, arm64)
#   ARCH_ALT: Used for Rust targets, manylinux suffix (e.g., x86_64, aarch64)
#
# Default values are for x86/amd64:
#   --build-arg ARCH=amd64 --build-arg ARCH_ALT=x86_64
#
# For arm64/aarch64, build with:
#   --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64
#TODO OPS-592: Leverage uname -m to determine ARCH instead of passing it as an arg
ARG ARCH={{ platform }}
ARG ARCH_ALT={{ "x86_64" if platform == "amd64" else "aarch64" }}

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF={{ context.dynamo_base.nixl_ucx_ref }}
ARG NIXL_REF={{ context.dynamo_base.nixl_ref }}
ARG NIXL_GDRCOPY_REF={{ context.dynamo_base.nixl_gdrcopy_ref }}

# Python configuration
ARG PYTHON_VERSION={{ context.dynamo_base.python_version }}
ARG CUDA_VERSION={{ context[framework].cuda_version }}

{%- if framework == "vllm" -%}
# Make sure to update the dependency version in pyproject.toml when updating this
ARG VLLM_REF={{ context.vllm.ref }}
# FlashInfer only respected when building vLLM from source, ie when VLLM_REF does not start with 'v' or for arm64 builds
ARG FLASHINF_REF={{ context.vllm.flashinf_ref }}
ARG TORCH_BACKEND={{ context.vllm[platform].torch_backend }}

# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""
{%- endif -%}

{% if framework == "sglang" %}
# Runtime image and build-time configuration (aligned with other backends)
# TODO: OPS-<number>: Use the same runtime image as the other backends
ARG SGLANG_RUNTIME_IMAGE={{ context.sglang.runtime_image }}
ARG SGLANG_RUNTIME_IMAGE_TAG={{ context.sglang.runtime_image_tag }}
ARG SGLANG_PYTHON_VERSION={{ context.sglang.python_version }}
{%- endif -%}

{% if framework == "trtllm" %}
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG PYTORCH_BASE_IMAGE={{ context.trtllm.pytorch_base_image }}
ARG PYTORCH_BASE_IMAGE_TAG={{ context.trtllm.pytorch_base_image_tag }}
ARG TRTLLM_RUNTIME_IMAGE={{ context.trtllm.runtime_image }}
ARG TRTLLM_RUNTIME_IMAGE_TAG={{ context.trtllm.runtime_image_tag }}

# TensorRT-LLM specific configuration
ARG HAS_TRTLLM_CONTEXT=0
ARG TENSORRTLLM_PIP_WHEEL={{ context.trtllm.pip_wheel }}
ARG TENSORRTLLM_INDEX_URL={{ context.trtllm.index_url }}
ARG GITHUB_TRTLLM_COMMIT

# Python configuration
ARG TRTLLM_PYTHON_VERSION={{ context[framework].python_version }}
{%- endif -%}
