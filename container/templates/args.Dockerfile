{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/args.Dockerfile ===
##########################
#### Build Arguments #####
##########################
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

# Python/CUDA configuration
ARG PYTHON_VERSION={{ context.dynamo.python_version }}
ARG CUDA_VERSION={{ cuda_version }}
ARG CUDA_MAJOR=${CUDA_VERSION%%.*}

# Base and runtime images configuration
{% set cuda_context_key = "cuda" + cuda_version %}
ARG BASE_IMAGE={{ context[framework].base_image }}
ARG BASE_IMAGE_TAG={{ context[framework][cuda_context_key].base_image_tag }}
{% if framework in ["sglang", "trtllm", "vllm"] -%}
ARG RUNTIME_IMAGE={{ context[framework].runtime_image }}
ARG RUNTIME_IMAGE_TAG={{ context[framework][cuda_context_key].runtime_image_tag }}
{%- endif %}

# Build configuration
ARG ENABLE_KVBM={{ context[framework].enable_kvbm }}
ARG CARGO_BUILD_JOBS

ARG NATS_VERSION={{ context.dynamo.nats_version }}
ARG ETCD_VERSION={{ context.dynamo.etcd_version }}

ARG ENABLE_MEDIA_FFMPEG={{ context[framework].enable_media_ffmpeg }}
ARG FFMPEG_VERSION={{ context.dynamo.ffmpeg_version }}
ARG ENABLE_GPU_MEMORY_SERVICE={{ context[framework].enable_gpu_memory_service }}

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF={{ context.dynamo.nixl_ucx_ref }}
ARG NIXL_REF={{ context.dynamo.nixl_ref }}
ARG NIXL_GDRCOPY_REF={{ context.dynamo.nixl_gdrcopy_ref }}
ARG NIXL_LIBFABRIC_REF={{ context.dynamo.nixl_libfabric_ref }}

{% if target == "dev" or target == "local-dev" %}
ARG FRAMEWORK={{ framework }}
{% endif %}

{% if target == "frontend" %}
ARG EPP_IMAGE={{ context.dynamo.epp_image }}
ARG FRONTEND_IMAGE={{ context.dynamo.frontend_image }}
{% endif %}

{% if framework == "vllm" -%}
# Make sure to update the dependency version in pyproject.toml when updating this
ARG VLLM_REF={{ context.vllm.vllm_ref }}
ARG MAX_JOBS={{ context.vllm.max_jobs }}
# FlashInfer only respected when building vLLM from source, ie when VLLM_REF does not start with 'v' or for arm64 builds
ARG FLASHINF_REF={{ context.vllm.flashinf_ref }}
ARG LMCACHE_REF={{ context.vllm.lmcache_ref }}

# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""

# ModelExpress for P2P weight transfer (optional)
ARG ENABLE_MODELEXPRESS_P2P={{ context.vllm.enable_modelexpress_p2p }}
ARG MODELEXPRESS_REF={{ context.vllm.modelexpress_ref }}
{%- endif -%}

{% if framework == "trtllm" %}
# TensorRT-LLM specific configuration
ARG HAS_TRTLLM_CONTEXT={{ context.trtllm.has_trtllm_context }}
ARG TENSORRTLLM_PIP_WHEEL={{ context.trtllm.pip_wheel }}
ARG TENSORRTLLM_INDEX_URL={{ context.trtllm.index_url }}
ARG GITHUB_TRTLLM_COMMIT={{ context.trtllm.github_trtllm_commit }}
ARG TRTLLM_WHEEL_IMAGE={{ context.trtllm.trtllm_wheel_image }}

# Copy pytorch installation from NGC PyTorch
ARG FLASHINFER_PYTHON_VER={{ context.trtllm.flashinfer_python_ver }}
ARG PYTORCH_TRITON_VER={{ context.trtllm.pytorch_triton_ver }}
ARG TORCHAO_VER={{ context.trtllm.torchao_ver }}
ARG TORCHDATA_VER={{ context.trtllm.torchdata_ver }}
ARG TORCHTITAN_VER={{ context.trtllm.torchtitan_ver }}
ARG TORCH_VER={{ context.trtllm.torch_version }}
ARG TORCH_TENSORRT_VER={{ context.trtllm.torch_tensorrt_version }}
ARG TORCHVISION_VER={{ context.trtllm.torchvision_version }}
ARG JINJA2_VER={{ context.trtllm.jinja2_version }}
ARG SYMPY_VER={{ context.trtllm.sympy_version }}
ARG FLASH_ATTN_VER={{ context.trtllm.flash_attn_version }}

# Python configuration
ARG TRTLLM_PYTHON_VERSION={{ context[framework].python_version }}
{%- endif -%}

{% if make_efa == true %}
ARG EFA_VERSION={{ context.dynamo.efa_version }}
ARG EFA_BASE_IMAGE={{ "runtime" if target=="runtime" else "dev" }}
{%- endif -%}