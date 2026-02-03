# Dynamo EPP Makefile
# Builds custom EPP image with Dynamo KV-aware routing plugins

# Image configuration
# Image lives in local cache only, not pushed to any registry
DOCKER_SERVER ?= dynamo
IMAGE_NAME := dynamo-epp
GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_TAG ?= $(shell git describe --tags --dirty --always 2>/dev/null || echo "dev")
IMAGE_REPO ?= $(DOCKER_SERVER)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)

# Build configuration
# Auto-detect host architecture for consistent builds with Dynamo library
# The Dynamo library is built for the host arch, so Docker must match
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
# Docker proxy for avoiding rate limits (e.g., ECR mirror)
# Set DOCKER_PROXY to prefix base images, e.g., DOCKER_PROXY=my-registry.com/dockerhub/
DOCKER_PROXY ?=

DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
BUILDER_IMAGE ?= $(DOCKER_PROXY)golang:1.24
BASE_IMAGE ?= $(DOCKER_PROXY)ubuntu:24.04

# Container tool
CONTAINER_TOOL ?= docker

# Kind cluster name for local testing
KIND_CLUSTER ?= kind

# Project directory
PROJECT_DIR := $(shell pwd)

# Dynamo directories
# Default: assume we're in dynamo/deploy/inference-gateway/epp
DYNAMO_DIR ?= $(shell cd $(PROJECT_DIR)/../../.. && pwd)
DYNAMO_LIB_DIR := $(PROJECT_DIR)/pkg/plugins/dynamo_kv_scorer/lib
DYNAMO_INCLUDE_DIR := $(PROJECT_DIR)/pkg/plugins/dynamo_kv_scorer/include

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Run go fmt
	go fmt ./...

.PHONY: vet
vet: ## Run go vet
	go vet ./...

.PHONY: tidy
tidy: ## Run go mod tidy
	go mod tidy

.PHONY: test
test: ## Run tests
	CGO_ENABLED=1 go test ./... -v

##@ Build

.PHONY: build
build: dynamo-lib-check ## Build the EPP binary locally (requires CGO and Dynamo libraries)
	CGO_ENABLED=1 go build -o bin/epp ./cmd/epp

.PHONY: build-with-lib
build-with-lib: dynamo-lib build ## Build Dynamo library and EPP binary

.PHONY: image-build
image-build: dynamo-lib-check ## Build the Docker image using buildx
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg COMMIT_SHA=$(GIT_COMMIT_SHA) \
		--build-arg BUILD_REF=$(GIT_TAG) \
		$(PUSH) \
		$(LOAD) \
		.

.PHONY: image-push
image-push: PUSH=--push ## Build and push the Docker image
image-push: image-build

.PHONY: image-load
image-load: LOAD=--load ## Build and load the Docker image locally
image-load: image-build

.PHONY: image-kind
image-kind: image-load ## Build and load the image into kind cluster
	kind load docker-image $(IMAGE_TAG) --name $(KIND_CLUSTER)

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

##@ Dynamo Library Build

.PHONY: dynamo-lib
dynamo-lib: ## Build Dynamo static library and copy to project
	@echo "Building Dynamo static library..."
	cd "$(DYNAMO_DIR)" && cargo build --release -p libdynamo_llm
	@echo "Generating C header..."
	@mkdir -p "$(DYNAMO_DIR)/lib/bindings/c/include/nvidia/dynamo_llm"
	cd "$(DYNAMO_DIR)" && \
		(cbindgen --config lib/bindings/c/cbindgen.toml --crate libdynamo_llm \
			--output lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h || \
		cp lib/bindings/c/src/fallback_header.h lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h)
	@echo "Copying files to EPP project..."
	@mkdir -p "$(DYNAMO_LIB_DIR)"
	@mkdir -p "$(DYNAMO_INCLUDE_DIR)"
	cp "$(DYNAMO_DIR)/lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h" "$(DYNAMO_INCLUDE_DIR)/"
	cp "$(DYNAMO_DIR)/target/release/libdynamo_llm_capi.a" "$(DYNAMO_LIB_DIR)/"
	@echo "Dynamo library ready!"

.PHONY: dynamo-lib-check
dynamo-lib-check: ## Check if Dynamo library files exist
	@if [ ! -f "$(DYNAMO_LIB_DIR)/libdynamo_llm_capi.a" ]; then \
		echo "ERROR: Dynamo library not found. Run 'make dynamo-lib' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(DYNAMO_INCLUDE_DIR)/llm_engine.h" ]; then \
		echo "ERROR: Dynamo header not found. Run 'make dynamo-lib' first."; \
		exit 1; \
	fi
	@echo "Dynamo library files found."

##@ Clean

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf bin/
	go clean

##@ All-in-one Build

.PHONY: all
all: dynamo-lib image-local-load ## Build Dynamo lib and Docker image, load locally

.PHONY: all-push
all-push: dynamo-lib image-push ## Build Dynamo lib and Docker image, push to registry

.PHONY: all-kind
all-kind: dynamo-lib image-kind ## Build Dynamo lib and Docker image, load to kind

##@ Info

.PHONY: info
info: ## Show build info
	@echo "Image Tag: $(IMAGE_TAG)"
	@echo "Git Commit: $(GIT_COMMIT_SHA)"
	@echo "Git Tag: $(GIT_TAG)"
	@echo "Platforms: $(PLATFORMS)"
	@echo "Docker Proxy: $(DOCKER_PROXY)"
	@echo "Builder Image: $(BUILDER_IMAGE)"
	@echo "Base Image: $(BASE_IMAGE)"
	@echo "Dynamo Dir: $(DYNAMO_DIR)"
	@echo "Dynamo Lib Dir: $(DYNAMO_LIB_DIR)"
	@echo "Dynamo Include Dir: $(DYNAMO_INCLUDE_DIR)"

