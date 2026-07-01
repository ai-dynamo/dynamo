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
{# device_key (e.g. "cuda12.9", "xpu") is provided by render.py so it
   propagates to every included template, not just this one. #}

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
{% if framework == "vllm" and device == "cuda" -%}
ARG VLLM_RUNTIME_BASE_IMAGE=${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG}
{% endif -%}
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
ARG NV_CODEC_HEADERS_REF={{ context.dynamo.nv_codec_headers_ref }}
ARG LIBVPX_REF={{ context.dynamo.libvpx_ref }}
{% if device == "cuda" -%}
ARG ENABLE_GPU_MEMORY_SERVICE={{ context[framework].enable_gpu_memory_service }}
{% endif %}

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_VERSION={{ context.dynamo.sccache_version }}
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF={{ context.dynamo.nixl_ucx_ref }}
{% if "nixl_ref" in context[framework].get(device_key, {}) -%}
ARG NIXL_REF={{ context[framework][device_key].nixl_ref }}
{% elif "nixl_ref" in context[framework] -%}
ARG NIXL_REF={{ context[framework].nixl_ref }}
{% endif -%}
{% if device == "cuda" %}
ARG NIXL_GDRCOPY_REF={{ context.dynamo.nixl_gdrcopy_ref }}
ARG NIXL_LIBFABRIC_REPO={{ context.dynamo.nixl_libfabric_repo }}
ARG NIXL_LIBFABRIC_REF={{ context.dynamo.nixl_libfabric_ref }}
ARG HWLOC_VERSION={{ context.dynamo.hwloc_version }}
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
ARG MAX_JOBS={{ context.vllm.max_jobs }}
{% if device == "cuda" -%}
# FlashInfer cubin/jit-cache version used by the vLLM installer.
ARG FLASHINF_REF={{ context.vllm.flashinf_ref }}
ARG FLASHINFER_GIT_URL={{ context.vllm.flashinfer_git_url }}
ARG FLASHINFER_GIT_REF={{ context.vllm.flashinfer_git_ref }}
ARG FLASHINFER_GIT_SHA={{ context.vllm.flashinfer_git_sha }}
ARG VLLM_GIT_URL={{ context.vllm.vllm_git_url }}
ARG VLLM_GIT_REF={{ context.vllm.vllm_git_ref }}
ARG VLLM_GIT_SHA={{ context.vllm.vllm_git_sha }}
ARG VLLM_INSTALL_MODE={{ context.vllm.vllm_install_mode }}
ARG VLLM_TORCH_VERSION={{ context.vllm.vllm_torch_version }}
ARG VLLM_TORCHVISION_VERSION={{ context.vllm.vllm_torchvision_version }}
ARG VLLM_TORCH_BACKEND={{ context.vllm.vllm_torch_backend }}
ARG VLLM_TORCH_CUDA_ARCH_LIST={{ context.vllm.vllm_torch_cuda_arch_list }}
ARG VLLM_PRECOMPILED_WHEEL_COMMIT={{ context.vllm.vllm_precompiled_wheel_commit }}
ARG VLLM_PRECOMPILED_WHEEL_VARIANT={{ context.vllm.vllm_precompiled_wheel_variant }}
ARG VLLM_NCCL_VERSION={{ context.vllm.vllm_nccl_version }}
ARG NCCL_CHECKPOINT_VERSION={{ context.vllm.nccl_checkpoint_version }}
{% endif %}
ARG VLLM_OMNI_REF={{ context.vllm.vllm_omni_ref }}

{% if device == "cuda" -%}
# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""

# aws-sdk-cpp tag for the NIXL OBJ / S3 backend (built in wheel_builder).
ARG AWS_SDK_CPP_VERSION={{ context.vllm.aws_sdk_cpp_version }}
{% endif %}
{%- endif -%}

{% if framework in ["vllm", "sglang"] -%}
# ModelExpress Python client for model loading (optional)
ARG MODELEXPRESS_VERSION={{ context[framework].modelexpress_version }}
{%- endif -%}

{% if framework == "sglang" and device == "xpu" -%}
# SGLang XPU build: clone and build from source (no pre-built runtime image)
ARG SGLANG_GIT_URL={{ context.sglang.xpu.sglang_git_url }}
ARG SGLANG_REF={{ context.sglang.xpu.sglang_ref }}
ARG SGLANG_KERNEL_GIT_URL={{ context.sglang.xpu.sglang_kernel_git_url }}
ARG SGLANG_KERNEL_REF={{ context.sglang.xpu.sglang_kernel_ref }}
{%- endif -%}

{% if make_efa == true %}
ARG EFA_VERSION={{ context.dynamo.efa_version }}
ARG EFA_BASE_IMAGE={{ "runtime" if target=="runtime" else "dev" }}
{%- endif -%}
