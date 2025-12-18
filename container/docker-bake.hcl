// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

variable "DOCKER_REGISTRY" {
    default = ""
}
variable "LOCAL_CACHE_DIR" {
  default = ""
  # Set this to a local directory path to enable local caching
}
variable "POSTFIX_TAG" {
  default = "latest"
}
variable "PREFIX_TAG" {
  default = "ai-dynamo/dynamo"
}
variable "SCCACHE_BUCKET" {
    default = ""
}
variable "SCCACHE_REGION" {
    default = ""
}
variable "USE_SCCACHE" {
    default = ""
}

target "_common" {
  args = {
    ARCH = "amd64"
    ARCH_ALT = "x86_64"
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base"
    BASE_IMAGE_TAG = "25.04-cuda12.9-devel-ubuntu24.04"
    ENABLE_KVBM = "true"
    ENABLE_MEDIA_NIXL = "false"
    NIXL_UCX_REF = "v1.19.0"
    NIXL_GDRCOPY_REF = "v2.5.1"
    NIXL_REF ="0.7.1"
    PYTHON_VERSION = "3.12"
    SCCACHE_BUCKET = "${SCCACHE_BUCKET}"
    SCCACHE_REGION = "${SCCACHE_REGION}"
    USE_SCCACHE = "${USE_SCCACHE}"
  }
  secret = concat(
    notequal("", USE_SCCACHE) ?
      [{
        id="aws-key-id"
        env="AWS_ACCESS_KEY_ID"
      }]
    : [],
    notequal("", USE_SCCACHE) ?
      [{
        id="aws-secret-id"
        env="AWS_SECRET_ACCESS_KEY"
      }]
    : []
  )
}

target "base" {
  dockerfile = "container/Dockerfile"
  target = "base"
  inherits = ["_common"]
  platforms = ["linux/amd64"]
  tags = [
    notequal("", DOCKER_REGISTRY) ? "${DOCKER_REGISTRY}:base-${POSTFIX_TAG}" : "${PREFIX_TAG}:base-${POSTFIX_TAG}"
  ]
  cache-from = concat(
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        src="${LOCAL_CACHE_DIR}/base-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:base-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:base-latest"
      }] : []
  )
  cache-to = concat(
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        mode="max",
        image-manifest=true,
        oci-mediatypes=true,
        ref="${DOCKER_REGISTRY}:base-${POSTFIX_TAG}"
      }] : [],
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        mode="max",
        dest="${LOCAL_CACHE_DIR}/base-${POSTFIX_TAG}"
      }] : []
  )
}

target "wheel_builder" {
  dockerfile = "container/Dockerfile"
  target = "wheel_builder"
  inherits = ["_common"]
  platforms = ["linux/amd64"]
  tags = [
    notequal("", DOCKER_REGISTRY) ? "${DOCKER_REGISTRY}:wheel_builder-${POSTFIX_TAG}" : "${PREFIX_TAG}:wheel_builder-${POSTFIX_TAG}"
  ]
  contexts = {
    base = "target:base"
  }
  cache-from = concat(
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        src="${LOCAL_CACHE_DIR}/wheel_builder-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:wheel_builder-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:wheel_builder-latest"
      }] : []
  )
  cache-to = concat(
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        mode="max",
        image-manifest=true,
        oci-mediatypes=true,
        ref="${DOCKER_REGISTRY}:wheel_builder-${POSTFIX_TAG}"
      }] : [],
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        mode="max",
        dest="${LOCAL_CACHE_DIR}/wheel_builder-${POSTFIX_TAG}"
      }] : []
  )
}

target "vllm-runtime" {
  dockerfile = "container/Dockerfile.vllm"
  target = "runtime"
  inherits = ["_common"]
  platforms  = ["linux/amd64"]
  tags = [
    notequal("", DOCKER_REGISTRY) ? "${DOCKER_REGISTRY}:vllm_runtime-${POSTFIX_TAG}" : "${PREFIX_TAG}:vllm_runtime-${POSTFIX_TAG}"
  ]
  contexts = {
    dynamo_base = "target:base"
    wheel_builder = "target:wheel_builder"
  }
  cache-from = concat(
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        src="${LOCAL_CACHE_DIR}/vllm_runtime-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:vllm_runtime-${POSTFIX_TAG}"
      }] : [],
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        ref="${DOCKER_REGISTRY}:vllm_runtime-latest"
      }] : []
  )
  cache-to = concat(
    notequal("",DOCKER_REGISTRY) ?
      [{
        type="registry",
        mode="max",
        image-manifest=true,
        oci-mediatypes=true,
        ref="${DOCKER_REGISTRY}:vllm_runtime-${POSTFIX_TAG}"
      }] : [],
    notequal("", LOCAL_CACHE_DIR) ?
      [{
        type="local",
        mode="max",
        dest="${LOCAL_CACHE_DIR}/vllm_runtime-${POSTFIX_TAG}"
      }] : []
  )
}