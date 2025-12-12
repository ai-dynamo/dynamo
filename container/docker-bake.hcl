variable "TAG" {
  default = "latest"
}

variable "LOCAL_CACHE_DIR" {
  default = "./docker-cache"
}

target "base" {
  dockerfile = "container/Dockerfile"
  target = "base"
  platforms = ["linux/amd64"]
  tags = ["dynamo/base:${TAG}"]
  cache-from = [
    "type=local,src=${LOCAL_CACHE_DIR}/base-latest",
    "type=local,src=${LOCAL_CACHE_DIR}/base-${TAG}"
  ]
  cache-to   = [ "type=local,dest=${LOCAL_CACHE_DIR}/base-${TAG},mode=max" ]
//   cache-from = [ "type=registry,ref=repo/app:cache" ]
//   cache-to = [ "type=registry,ref=repo/app:cache,mode=max" ]
  args = {
    ARCH = "amd64"
    ARCH_ALT = "x86_64"
    NIXL_REF ="0.7.1"
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base"
    BASE_IMAGE_TAG = "25.04-cuda12.9-devel-ubuntu24.04"
    NIXL_UCX_REF = "v1.19.0"
    NIXL_GDRCOPY_REF = "v2.5.1"
    PYTHON_VERSION = "3.12"
    ENABLE_KVBM = "true"
    USE_SCCACHE = ""
    SCCACHE_BUCKET = ""
    SCCACHE_REGION = ""
    ENABLE_MEDIA_NIXL = "false"
  }
}

target "wheel_builder" {
  dockerfile = "container/Dockerfile"
  target = "wheel_builder"
  platforms = ["linux/amd64"]
  tags = ["dynamo/wheel_builder:${TAG}"]
  cache-from = [
    "type=local,src=${LOCAL_CACHE_DIR}/base-latest",
    "type=local,src=${LOCAL_CACHE_DIR}/base-${TAG}",
    "type=local,src=${LOCAL_CACHE_DIR}/wheel_builder-${TAG}"
  ]
  cache-to   = [ "type=local,dest=${LOCAL_CACHE_DIR}/wheel_builder-${TAG},mode=max" ]
  contexts = {
    base = "target:base"
  }
  args = {
    ARCH = "amd64"
    ARCH_ALT = "x86_64"
    NIXL_REF ="0.7.1"
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base"
    BASE_IMAGE_TAG = "25.04-cuda12.9-devel-ubuntu24.04"
    NIXL_UCX_REF = "v1.19.0"
    NIXL_GDRCOPY_REF = "v2.5.1"
    PYTHON_VERSION = "3.12"
    ENABLE_KVBM = "true"
    USE_SCCACHE = ""
    SCCACHE_BUCKET = ""
    SCCACHE_REGION = ""
    ENABLE_MEDIA_NIXL = "false"
  }
}

target "vllm-runtime" {
  dockerfile = "container/Dockerfile.vllm"
  target = "runtime"
  platforms  = ["linux/amd64"]
  tags = ["dynamo/vllm-runtime:${TAG}"]
  cache-from = [
    "type=local,src=${LOCAL_CACHE_DIR}/base-latest",
    "type=local,src=${LOCAL_CACHE_DIR}/base-${TAG}",
    "type=local,src=${LOCAL_CACHE_DIR}/wheel_builder-${TAG}",
    "type=local,src=${LOCAL_CACHE_DIR}/vllm-runtime-${TAG}"
  ]
  cache-to = [ "type=local,dest=${LOCAL_CACHE_DIR}/vllm-runtime-${TAG},mode=max" ]
  contexts = {
    dynamo_base = "target:base"
    wheel_builder = "target:wheel_builder"
  }
  args = {
    ARCH = "amd64"
    ARCH_ALT = "x86_64"
    NIXL_REF ="0.7.1"
    BASE_IMAGE = "nvcr.io/nvidia/cuda-dl-base"
    BASE_IMAGE_TAG = "25.04-cuda12.9-devel-ubuntu24.04"
    NIXL_UCX_REF = "v1.19.0"
    NIXL_GDRCOPY_REF = "v2.5.1"
    PYTHON_VERSION = "3.12"
    ENABLE_KVBM = "true"
    USE_SCCACHE = ""
    SCCACHE_BUCKET = ""
    SCCACHE_REGION = ""
    ENABLE_MEDIA_NIXL = "false"
  }
}