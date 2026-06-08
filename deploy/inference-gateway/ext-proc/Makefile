# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo Rust EPP Makefile
# Builds the pure-Rust ext_proc EPP image (no Go, no CGO)

# Image configuration
DOCKER_SERVER ?= dynamo
IMAGE_NAME := dynamo-rust-epp
GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_TAG ?= $(shell git describe --tags --dirty --always 2>/dev/null || echo "dev")
IMAGE_REPO ?= $(DOCKER_SERVER)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)

# Build configuration
HOST_ARCH := $(shell uname -m)
ifeq ($(HOST_ARCH),x86_64)
    PLATFORMS ?= linux/amd64
else ifeq ($(HOST_ARCH),aarch64)
    PLATFORMS ?= linux/arm64
else ifeq ($(HOST_ARCH),arm64)
    PLATFORMS ?= linux/arm64
else
    PLATFORMS ?= linux/amd64
endif
MULTIARCH_PLATFORMS ?= linux/amd64,linux/arm64
DOCKER_PROXY ?=
EXTRA_BUILD_ARGS ?=

# sccache configuration (CI only)
USE_SCCACHE ?=
SCCACHE_BUCKET ?=
SCCACHE_REGION ?=

DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
RUST_IMAGE ?= $(DOCKER_PROXY)rust:1.93.1
BASE_IMAGE ?= $(DOCKER_PROXY)ubuntu:24.04

# Container tool
CONTAINER_TOOL ?= docker

# Kind cluster name for local testing
KIND_CLUSTER ?= kind

# Project directory
PROJECT_DIR := $(shell pwd)

# Dynamo repository root (build context for Docker)
DYNAMO_DIR ?= $(shell cd $(PROJECT_DIR)/../../.. && pwd)

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Run cargo fmt
	cd "$(DYNAMO_DIR)" && cargo fmt -p dynamo-ext-proc

.PHONY: check
check: ## Run cargo check
	cd "$(DYNAMO_DIR)" && cargo check -p dynamo-ext-proc

.PHONY: clippy
clippy: ## Run clippy lints
	cd "$(DYNAMO_DIR)" && cargo clippy -p dynamo-ext-proc --no-deps --all-targets -- -D warnings

.PHONY: test
test: ## Run tests
	cd "$(DYNAMO_DIR)" && cargo test -p dynamo-ext-proc

##@ Build

.PHONY: build
build: ## Build the EPP binary locally
	cd "$(DYNAMO_DIR)" && cargo build --release -p dynamo-ext-proc

.PHONY: build-debug
build-debug: ## Build the EPP binary locally (debug mode)
	cd "$(DYNAMO_DIR)" && cargo build -p dynamo-ext-proc

.PHONY: image-build
image-build: ## Build the Docker image
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-context dynamo=$(DYNAMO_DIR) \
		--build-arg RUST_IMAGE=$(RUST_IMAGE) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		$(EXTRA_BUILD_ARGS) $(PUSH) $(LOAD) .

.PHONY: image-push
image-push: PUSH=--push ## Build and push the Docker image
image-push: image-build

.PHONY: image-load
image-load: LOAD=--load ## Build and load the Docker image locally
image-load: image-build

.PHONY: image-kind
image-kind: image-load ## Build and load the image into kind cluster
	kind load docker-image $(IMAGE_TAG) --name $(KIND_CLUSTER)

##@ Multi-Architecture Builds

SCCACHE_ARGS =
ifeq ($(USE_SCCACHE),true)
  SCCACHE_ARGS += --build-arg USE_SCCACHE=true
  SCCACHE_ARGS += --build-arg SCCACHE_BUCKET=$(SCCACHE_BUCKET)
  SCCACHE_ARGS += --build-arg SCCACHE_REGION=$(SCCACHE_REGION)
  ifneq ($(AWS_WEB_IDENTITY_TOKEN_FILE),)
    ifneq ($(AWS_ROLE_ARN),)
      SCCACHE_ARGS += --secret id=aws-web-identity-token,src=$(AWS_WEB_IDENTITY_TOKEN_FILE)
      SCCACHE_ARGS += --secret id=aws-role-arn,env=AWS_ROLE_ARN
    endif
  endif
endif

.PHONY: image-multiarch
image-multiarch: ## Build multi-arch image (requires --push)
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(MULTIARCH_PLATFORMS) \
		--build-context dynamo=$(DYNAMO_DIR) \
		--build-arg RUST_IMAGE=$(RUST_IMAGE) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		$(SCCACHE_ARGS) $(EXTRA_BUILD_ARGS) $(PUSH) .

.PHONY: image-multiarch-push
image-multiarch-push: PUSH=--push ## Build and push multi-arch image
image-multiarch-push: image-multiarch

##@ Local Development with Buildx

.PHONY: image-local-build
image-local-build: ## Build image using a new buildx builder
	BUILDER=$$($(DOCKER_BUILDX_CMD) create --use) && \
	$(MAKE) image-build PUSH=$(PUSH) LOAD=$(LOAD) && \
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: image-local-push
image-local-push: PUSH=--push ## Build and push using local buildx builder
image-local-push: image-local-build

.PHONY: image-local-load
image-local-load: LOAD=--load ## Build and load using local buildx builder
image-local-load: image-local-build

##@ Clean

.PHONY: clean
clean: ## Clean build artifacts
	cd "$(DYNAMO_DIR)" && cargo clean -p dynamo-ext-proc

##@ Info

.PHONY: info
info: ## Show build info
	@echo "Image Tag:     $(IMAGE_TAG)"
	@echo "Git Commit:    $(GIT_COMMIT_SHA)"
	@echo "Git Tag:       $(GIT_TAG)"
	@echo "Platforms:     $(PLATFORMS)"
	@echo "Rust Image:    $(RUST_IMAGE)"
	@echo "Base Image:    $(BASE_IMAGE)"
	@echo "Dynamo Dir:    $(DYNAMO_DIR)"
