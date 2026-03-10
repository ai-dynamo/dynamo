# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image URL to use all building/pushing image targets
IMG ?= nvcr.io/nvidian/dynamo-dev/snapshot-agent:latest
PLACEHOLDER_IMG ?= nvcr.io/nvidian/dynamo-dev/dynamo-vllm-placeholder:latest
# PLACEHOLDER_BASE_IMG must be provided when building placeholder (no default)

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# CONTAINER_TOOL defines the container tool to be used for building images.
CONTAINER_TOOL ?= docker
# Snapshot runtime images ship cuda-checkpoint and are amd64-only.
RUNTIME_IMAGE_PLATFORM ?= linux/amd64

# Setting SHELL to bash allows bash commands to be executed by recipes.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: test
test: fmt vet ## Run tests.
	go test ./... -v -coverprofile cover.out

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter.
	$(GOLANGCI_LINT) run

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes.
	$(GOLANGCI_LINT) run --fix

##@ Build

.PHONY: build
build: fmt vet ## Build snapshot-agent binary.
	go build -o bin/snapshot-agent ./cmd/agent

##@ Docker

.PHONY: docker-build-agent
docker-build-agent: ## Build snapshot-agent docker image (linux/amd64 only).
	$(CONTAINER_TOOL) build --platform ${RUNTIME_IMAGE_PLATFORM} --target agent -t ${IMG} .

.PHONY: docker-build-placeholder
docker-build-placeholder: ## Build placeholder image for checkpoint restore (linux/amd64 only). Requires PLACEHOLDER_BASE_IMG.
ifndef PLACEHOLDER_BASE_IMG
	$(error PLACEHOLDER_BASE_IMG is required. Example: make docker-build-placeholder PLACEHOLDER_BASE_IMG=nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1-cuda13)
endif
	@BASE_IMAGE_USER="$$( $(CONTAINER_TOOL) image inspect --format '{{.Config.User}}' ${PLACEHOLDER_BASE_IMG} 2>/dev/null || true )"; \
	if [ -z "$$BASE_IMAGE_USER" ]; then \
		$(CONTAINER_TOOL) pull ${PLACEHOLDER_BASE_IMG} >/dev/null; \
		BASE_IMAGE_USER="$$( $(CONTAINER_TOOL) image inspect --format '{{.Config.User}}' ${PLACEHOLDER_BASE_IMG} 2>/dev/null || true )"; \
	fi; \
	if [ -z "$$BASE_IMAGE_USER" ]; then BASE_IMAGE_USER=root; fi; \
	$(CONTAINER_TOOL) build --platform ${RUNTIME_IMAGE_PLATFORM} --target placeholder \
		--build-arg BASE_IMAGE=${PLACEHOLDER_BASE_IMG} \
		--build-arg BASE_IMAGE_USER=$$BASE_IMAGE_USER \
		-t ${PLACEHOLDER_IMG} .

.PHONY: docker-push-agent
docker-push-agent: ## Push snapshot-agent docker image.
	$(CONTAINER_TOOL) push ${IMG}

.PHONY: docker-push-placeholder
docker-push-placeholder: ## Push placeholder docker image.
	$(CONTAINER_TOOL) push ${PLACEHOLDER_IMG}

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint-$(GOLANGCI_LINT_VERSION)

## Tool Versions
GOLANGCI_LINT_VERSION ?= v1.62.2

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/cmd/golangci-lint,${GOLANGCI_LINT_VERSION})

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary (ideally with version)
# $2 - package url which can be installed
# $3 - specific version of package
define go-install-tool
@[ -f $(1) ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv "$$(echo "$(1)" | sed "s/-$(3)$$//")" $(1) ;\
}
endef

.PHONY: coverage
coverage: test
	go tool cover -func=cover.out
