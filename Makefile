# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# Top-level Makefile for building Dynamo container images and running tests.
#
# Usage:
#   make docker-build-planner REGISTRY=jont828               # build planner
#   make docker-build-trtllm  REGISTRY=jont828               # build trtllm runtime
#   make docker-push-planner  REGISTRY=jont828               # push planner
#   make docker-build-all     REGISTRY=myregistry TAG=dev    # build everything
#   make test-e2e-dgdr DGDR_IMAGE=jont828/dynamo-planner:latest DGDR_TEST_FLAGS='...'
# ============================================================================

# --- Configurable variables -------------------------------------------------

# Container registry prefix (e.g. "jont828", "nvcr.io/nvidia/ai-dynamo").
# When set, images are tagged as REGISTRY/IMAGE_NAME:TAG.
REGISTRY ?=

# Image tag applied to all built images.
TAG ?= latest

# Container build tool (docker or podman).
CONTAINER_TOOL ?= docker

# CUDA versions used by render.py. Override for non-default builds.
# Valid values are constrained by container/render.py (currently 13.0 / 13.1).
CUDA_VERSION ?= 13.0
CUDA_VERSION_TRTLLM ?= 13.1
# CUDA version for the dynamo-framework images (planner, frontend).
CUDA_VERSION_DYNAMO ?= 13.0

# Platform passed to render.py (not used for multi-arch buildx here).
PLATFORM ?= linux/amd64

# Make-level build parallelism. Each docker-build-* target is a heavy container
# build, so running several at once under `make -j` can exhaust RAM/disk. By
# default we force serial execution even when -j is passed. Set
# PARALLEL_BUILDS=1 on a machine with ample resources to allow concurrent builds
# (Option B gives every target its own rendered Dockerfile, so this is safe).
#   make docker-build-all                        -> serial
#   make -j docker-build-all                     -> serial (default)
#   make -j docker-build-all PARALLEL_BUILDS=1   -> concurrent
PARALLEL_BUILDS ?=
ifndef PARALLEL_BUILDS
.NOTPARALLEL:
endif

# --- Derived image names ----------------------------------------------------

_prefix = $(if $(REGISTRY),$(REGISTRY)/,)

PLANNER_IMG  = $(_prefix)dynamo-planner:$(TAG)
FRONTEND_IMG = $(_prefix)dynamo-frontend:$(TAG)
VLLM_IMG     = $(_prefix)vllm-runtime:$(TAG)
TRTLLM_IMG   = $(_prefix)tensorrtllm-runtime:$(TAG)
SGLANG_IMG   = $(_prefix)sglang-runtime:$(TAG)
OPERATOR_IMG = $(_prefix)dynamo-operator:$(TAG)

# --- Rendering helper -------------------------------------------------------
# render.py generates a Dockerfile from Jinja2 templates. Without
# --output-short-filename it emits a unique path per framework/target/arch:
#   container/<framework>-<target>-cuda<cuda_version>-<arch>-rendered.Dockerfile
# Using unique names (instead of the shared container/rendered.Dockerfile) keeps
# the docker-build-* targets safe under parallel make: each build reads its own
# rendered Dockerfile and cannot clobber another target's. All such files match
# *rendered.Dockerfile in .gitignore.
RENDER = python3 container/render.py --platform $(PLATFORM)

# Architecture component of the rendered filename: render.py strips the
# "linux/" prefix from --platform (e.g. linux/amd64 -> amd64).
_arch = $(lastword $(subst /, ,$(PLATFORM)))

# rendered_dockerfile(framework,target,cuda_version) -> path render.py writes to.
rendered_dockerfile = container/$(1)-$(2)-cuda$(3)-$(_arch)-rendered.Dockerfile

PLANNER_DOCKERFILE  = $(call rendered_dockerfile,dynamo,planner,$(CUDA_VERSION_DYNAMO))
FRONTEND_DOCKERFILE = $(call rendered_dockerfile,dynamo,frontend,$(CUDA_VERSION_DYNAMO))
VLLM_DOCKERFILE     = $(call rendered_dockerfile,vllm,runtime,$(CUDA_VERSION))
TRTLLM_DOCKERFILE   = $(call rendered_dockerfile,trtllm,runtime,$(CUDA_VERSION_TRTLLM))
SGLANG_DOCKERFILE   = $(call rendered_dockerfile,sglang,runtime,$(CUDA_VERSION))

# ============================================================================
# Docker build targets
# ============================================================================

.PHONY: docker-build-planner
docker-build-planner: ## Build the planner image.
	$(RENDER) --framework dynamo --target planner --cuda-version $(CUDA_VERSION_DYNAMO)
	$(CONTAINER_TOOL) build -t $(PLANNER_IMG) -f $(PLANNER_DOCKERFILE) .

.PHONY: docker-build-frontend
docker-build-frontend: ## Build the frontend image.
	$(RENDER) --framework dynamo --target frontend --cuda-version $(CUDA_VERSION_DYNAMO)
	$(CONTAINER_TOOL) build -t $(FRONTEND_IMG) -f $(FRONTEND_DOCKERFILE) .

.PHONY: docker-build-vllm
docker-build-vllm: ## Build the vLLM runtime image.
	$(RENDER) --framework vllm --target runtime --cuda-version $(CUDA_VERSION)
	$(CONTAINER_TOOL) build -t $(VLLM_IMG) -f $(VLLM_DOCKERFILE) .

.PHONY: docker-build-trtllm
docker-build-trtllm: ## Build the TensorRT-LLM runtime image.
	$(RENDER) --framework trtllm --target runtime --cuda-version $(CUDA_VERSION_TRTLLM)
	$(CONTAINER_TOOL) build -t $(TRTLLM_IMG) -f $(TRTLLM_DOCKERFILE) .

.PHONY: docker-build-sglang
docker-build-sglang: ## Build the SGLang runtime image.
	$(RENDER) --framework sglang --target runtime --cuda-version $(CUDA_VERSION)
	$(CONTAINER_TOOL) build -t $(SGLANG_IMG) -f $(SGLANG_DOCKERFILE) .

.PHONY: docker-build-operator
docker-build-operator: ## Build the operator image (proxies to deploy/operator).
	$(MAKE) -C deploy/operator docker-build IMG=$(OPERATOR_IMG)

.PHONY: docker-build-all
docker-build-all: docker-build-planner docker-build-frontend docker-build-vllm docker-build-trtllm docker-build-sglang docker-build-operator ## Build all images.

# ============================================================================
# Docker push targets
# ============================================================================

.PHONY: docker-push-planner
docker-push-planner: ## Push the planner image.
	$(CONTAINER_TOOL) push $(PLANNER_IMG)

.PHONY: docker-push-frontend
docker-push-frontend: ## Push the frontend image.
	$(CONTAINER_TOOL) push $(FRONTEND_IMG)

.PHONY: docker-push-vllm
docker-push-vllm: ## Push the vLLM runtime image.
	$(CONTAINER_TOOL) push $(VLLM_IMG)

.PHONY: docker-push-trtllm
docker-push-trtllm: ## Push the TensorRT-LLM runtime image.
	$(CONTAINER_TOOL) push $(TRTLLM_IMG)

.PHONY: docker-push-sglang
docker-push-sglang: ## Push the SGLang runtime image.
	$(CONTAINER_TOOL) push $(SGLANG_IMG)

.PHONY: docker-push-operator
docker-push-operator: ## Push the operator image.
	$(CONTAINER_TOOL) push $(OPERATOR_IMG)

.PHONY: docker-push-all
docker-push-all: docker-push-planner docker-push-frontend docker-push-vllm docker-push-trtllm docker-push-sglang docker-push-operator ## Push all images.

# ============================================================================
# Test targets (proxy to deploy/operator/Makefile)
# ============================================================================

.PHONY: test-e2e-dgdr
test-e2e-dgdr: ## Run DGDR e2e tests (requires DGDR_IMAGE).
	$(MAKE) -C deploy/operator test-e2e-dgdr DGDR_IMAGE=$(DGDR_IMAGE) DGDR_TEST_FLAGS='$(DGDR_TEST_FLAGS)'

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m [VAR=value ...]\n\nVariables:\n  REGISTRY          Container registry prefix (e.g. jont828)\n  TAG               Image tag (default: latest)\n  CONTAINER_TOOL    docker or podman (default: docker)\n  CUDA_VERSION      CUDA version for vllm/sglang (default: 13.0)\n  CUDA_VERSION_TRTLLM  CUDA version for trtllm (default: 13.1)\n\nTargets:\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
