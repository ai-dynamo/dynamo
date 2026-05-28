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
CUDA_VERSION ?= 12.9
CUDA_VERSION_TRTLLM ?= 13.1

# Platform passed to render.py (not used for multi-arch buildx here).
PLATFORM ?= linux/amd64

# --- Derived image names ----------------------------------------------------

_prefix = $(if $(REGISTRY),$(REGISTRY)/,)

PLANNER_IMG  = $(_prefix)dynamo-planner:$(TAG)
FRONTEND_IMG = $(_prefix)dynamo-frontend:$(TAG)
VLLM_IMG     = $(_prefix)vllm-runtime:$(TAG)
TRTLLM_IMG   = $(_prefix)tensorrtllm-runtime:$(TAG)
SGLANG_IMG   = $(_prefix)sglang-runtime:$(TAG)
OPERATOR_IMG = $(_prefix)dynamo-operator:$(TAG)

# --- Rendering helper -------------------------------------------------------
# render.py generates container/rendered.Dockerfile from Jinja2 templates.
RENDER = python3 container/render.py --output-short-filename --platform $(PLATFORM)

# ============================================================================
# Docker build targets
# ============================================================================

.PHONY: docker-build-planner
docker-build-planner: ## Build the planner image.
	$(RENDER) --framework dynamo --target planner
	$(CONTAINER_TOOL) build -t $(PLANNER_IMG) -f container/rendered.Dockerfile .

.PHONY: docker-build-frontend
docker-build-frontend: ## Build the frontend image.
	$(RENDER) --framework dynamo --target frontend
	$(CONTAINER_TOOL) build -t $(FRONTEND_IMG) -f container/rendered.Dockerfile .

.PHONY: docker-build-vllm
docker-build-vllm: ## Build the vLLM runtime image.
	$(RENDER) --framework vllm --target runtime --cuda-version $(CUDA_VERSION)
	$(CONTAINER_TOOL) build -t $(VLLM_IMG) -f container/rendered.Dockerfile .

.PHONY: docker-build-trtllm
docker-build-trtllm: ## Build the TensorRT-LLM runtime image.
	$(RENDER) --framework trtllm --target runtime --cuda-version $(CUDA_VERSION_TRTLLM)
	$(CONTAINER_TOOL) build -t $(TRTLLM_IMG) -f container/rendered.Dockerfile .

.PHONY: docker-build-sglang
docker-build-sglang: ## Build the SGLang runtime image.
	$(RENDER) --framework sglang --target runtime --cuda-version $(CUDA_VERSION)
	$(CONTAINER_TOOL) build -t $(SGLANG_IMG) -f container/rendered.Dockerfile .

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
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m [VAR=value ...]\n\nVariables:\n  REGISTRY          Container registry prefix (e.g. jont828)\n  TAG               Image tag (default: latest)\n  CONTAINER_TOOL    docker or podman (default: docker)\n  CUDA_VERSION      CUDA version for vllm/sglang (default: 12.9)\n  CUDA_VERSION_TRTLLM  CUDA version for trtllm (default: 13.1)\n\nTargets:\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
