{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/args.Dockerfile ===
##########################
#### Build Arguments #####
##########################
# TARGETARCH is set automatically by Docker BuildKit for every --platform build.
# It must NOT be declared in the global scope (before any FROM) — doing so shadows
# the automatic per-platform value that BuildKit injects.
#
# In each stage that needs it, re-declare with:  ARG TARGETARCH
#
# ARCH_ALT (x86_64 / aarch64) is computed inline in RUN steps:
#   ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64")
ARG DEVICE={{ device }}
{% if device == "cuda" -%}
{% set device_key = device + cuda_version -%}
{% else -%}
{% set device_key = device -%}
{% endif %}

# Python/CUDA configuration
ARG PYTHON_VERSION={{ context.dynamo.python_version }}
{% if device == "cuda" -%}
ARG CUDA_VERSION={{ cuda_version }}
ARG CUDA_MAJOR=${CUDA_VERSION%%.*}
{% endif %}

# Base and runtime images configuration
ARG BASE_IMAGE={{ context[framework][device_key].base_image }}
ARG BASE_IMAGE_TAG={{ context[framework][device_key].base_image_tag }}
{% if framework in ["sglang", "trtllm", "vllm"] -%}
ARG RUNTIME_IMAGE={{ context[framework][device_key].runtime_image }}
ARG RUNTIME_IMAGE_TAG={{ context[framework][device_key].runtime_image_tag }}
{%- endif %}

# wheel builder image selection
{% if device == "xpu" or device == "cpu" %}
ARG WHEEL_BUILDER_IMAGE=${BASE_IMAGE}:${BASE_IMAGE_TAG}
{% elif platform == "multi" %}
{# Multi-arch: manylinux selection is handled via --platform-pinned stage aliases   #}
{# in wheel_builder.Dockerfile using TARGETARCH. No static ARG needed here.         #}
{% else %}
ARG WHEEL_BUILDER_IMAGE=quay.io/pypa/manylinux_2_28_{{ "x86_64" if platform == "amd64" else "aarch64" }}
{% endif %}

# Build configuration
ARG ENABLE_KVBM={{ context[framework].enable_kvbm }}
ARG CARGO_BUILD_JOBS

ARG NATS_VERSION={{ context.dynamo.nats_version }}
ARG ETCD_VERSION={{ context.dynamo.etcd_version }}

ARG ENABLE_MEDIA_FFMPEG={{ context[framework].enable_media_ffmpeg }}
ARG FFMPEG_VERSION={{ context.dynamo.ffmpeg_version }}
{% if device == "cuda" -%}
ARG ENABLE_GPU_MEMORY_SERVICE={{ context[framework].enable_gpu_memory_service }}
{% endif %}

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF={{ context.dynamo.nixl_ucx_ref }}
{% if "nixl_ref" in context[framework] -%}
ARG NIXL_REF={{ context[framework].nixl_ref }}
{% endif -%}
{% if device == "cuda" %}
ARG NIXL_GDRCOPY_REF={{ context.dynamo.nixl_gdrcopy_ref }}
ARG NIXL_LIBFABRIC_REF={{ context.dynamo.nixl_libfabric_ref }}
{% endif %}

{% if target == "dev" or target == "local-dev" %}
ARG FRAMEWORK={{ framework }}
{% endif %}

{% if target == "frontend" %}
ARG EPP_IMAGE={{ context.dynamo.epp_image }}
ARG FRONTEND_IMAGE={{ context.dynamo.frontend_image }}
{% endif %}

{% if target == "planner" %}
ARG PLANNER_BUILD_IMAGE={{ context.dynamo.planner_build_image }}
ARG PLANNER_BUILD_IMAGE_TAG={{ context.dynamo.planner_build_image_tag }}
ARG PLANNER_RUNTIME_IMAGE={{ context.dynamo.planner_runtime_image }}
ARG PLANNER_RUNTIME_IMAGE_TAG={{ context.dynamo.planner_runtime_image_tag }}
{% endif %}

{% if framework == "vllm" -%}
# Make sure to update the dependency version in pyproject.toml when updating this
ARG VLLM_REF={{ context[framework][device_key].vllm_ref }}
ARG MAX_JOBS={{ context.vllm.max_jobs }}
{% if device == "cuda" -%}
# FlashInfer cubin/jit-cache version used by the vLLM installer.
ARG FLASHINF_REF={{ context.vllm.flashinf_ref }}
{% endif %}
ARG LMCACHE_REF={{ context.vllm.lmcache_ref }}
ARG VLLM_OMNI_REF={{ context.vllm.vllm_omni_ref }}

{% if device == "cuda" -%}
# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""

# ModelExpress for P2P weight transfer (optional)
ARG ENABLE_MODELEXPRESS_P2P={{ context.vllm.enable_modelexpress_p2p }}
ARG MODELEXPRESS_REF={{ context.vllm.modelexpress_ref }}
{% endif %}
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
