# Dynamo EPP Makefile
# Builds custom EPP image with Dynamo KV-aware routing plugins

# Image configuration
DOCKER_SERVER ?= dynamo
IMAGE_NAME := dynamo-epp
GIT_COMMIT_SHA ?= $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_TAG ?= $(shell git describe --tags --dirty --always 2>/dev/null || echo "dev")
IMAGE_REPO ?= $(DOCKER_SERVER)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)

# Build configuration
# PLATFORMS: defaults to host arch for fast local builds (image-load, image-kind)
# MULTIARCH_PLATFORMS: used by image-multiarch for multi-architecture manifests
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
# Docker proxy for avoiding rate limits (e.g., ECR mirror)
DOCKER_PROXY ?=

DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
RUST_IMAGE ?= $(DOCKER_PROXY)rust:1.90
BUILDER_IMAGE ?= $(DOCKER_PROXY)golang:1.24
BASE_IMAGE ?= $(DOCKER_PROXY)ubuntu:24.04

# Container tool
CONTAINER_TOOL ?= docker

# Kind cluster name for local testing
KIND_CLUSTER ?= kind

# Project directory
PROJECT_DIR := $(shell pwd)

# Dynamo repository root (build context for Docker)
DYNAMO_DIR ?= $(shell cd $(PROJECT_DIR)/../../.. && pwd)

# Local development paths (for host-native builds only, not used by Docker)
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
image-build: ## Build the Docker image (self-contained, no host prerequisites)
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-context dynamo=$(DYNAMO_DIR) \
		--build-arg RUST_IMAGE=$(RUST_IMAGE) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg COMMIT_SHA=$(GIT_COMMIT_SHA) \
		--build-arg BUILD_REF=$(GIT_TAG) \
		$(PUSH) $(LOAD) .

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

.PHONY: image-multiarch
image-multiarch: ## Build multi-arch image (requires --push ; --load not supported)
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(MULTIARCH_PLATFORMS) \
		--build-context dynamo=$(DYNAMO_DIR) \
		--build-arg RUST_IMAGE=$(RUST_IMAGE) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg COMMIT_SHA=$(GIT_COMMIT_SHA) \
		--build-arg BUILD_REF=$(GIT_TAG) \
		$(PUSH) .

.PHONY: image-multiarch-push
image-multiarch-push: PUSH=--push ## Build and push multi-arch image to registry
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

##@ Dynamo Library (for local host-native builds only)

.PHONY: dynamo-lib
dynamo-lib: ## Build Dynamo static library and copy to project (for local dev)
	@echo "Building Dynamo static library..."
	cd "$(DYNAMO_DIR)" && cargo build --release -p libdynamo_llm
	@echo "Copying files to EPP project..."
	@mkdir -p "$(DYNAMO_LIB_DIR)"
	@mkdir -p "$(DYNAMO_INCLUDE_DIR)"
	cp "$(DYNAMO_DIR)/lib/bindings/c/include/nvidia/dynamo_llm/llm_engine.h" "$(DYNAMO_INCLUDE_DIR)/"
	cp "$(DYNAMO_DIR)/target/release/libdynamo_llm_capi.a" "$(DYNAMO_LIB_DIR)/"
	@echo "Dynamo library ready!"

.PHONY: dynamo-lib-check
dynamo-lib-check: ## Check if Dynamo library files exist (for local dev)
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
all: image-local-load ## Build Docker image and load locally

.PHONY: all-push
all-push: image-push ## Build Docker image and push to registry

.PHONY: all-kind
all-kind: image-kind ## Build Docker image and load to kind

##@ Info

.PHONY: info
info: ## Show build info
	@echo "Image Tag: $(IMAGE_TAG)"
	@echo "Git Commit: $(GIT_COMMIT_SHA)"
	@echo "Git Tag: $(GIT_TAG)"
	@echo "Platforms: $(PLATFORMS)"
	@echo "Multi-Arch Platforms: $(MULTIARCH_PLATFORMS)"
	@echo "Docker Proxy: $(DOCKER_PROXY)"
	@echo "Rust Image: $(RUST_IMAGE)"
	@echo "Builder Image: $(BUILDER_IMAGE)"
	@echo "Base Image: $(BASE_IMAGE)"
	@echo "Dynamo Dir: $(DYNAMO_DIR)"
