# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image URL to use all building/pushing image targets
IMG ?= nvcr.io/nvidian/dynamo-dev/chrek-agent:latest
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
	$(GOLANGCI_LINT) run --timeout=5m

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes.
	$(GOLANGCI_LINT) run --fix --timeout=5m

##@ Build

.PHONY: build
build: fmt vet ## Build chrek-agent and restore-entrypoint binaries.
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/chrek-agent ./cmd/agent
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/restore-entrypoint ./cmd/restore-entrypoint

.PHONY: build-agent
build-agent: fmt vet ## Build chrek-agent binary only.
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/chrek-agent ./cmd/agent

.PHONY: build-restore
build-restore: fmt vet ## Build restore-entrypoint binary only.
	CGO_ENABLED=0 go build -ldflags="-w -s" -o bin/restore-entrypoint ./cmd/restore-entrypoint

.PHONY: run
run: build ## Run chrek-agent from your host.
	./bin/chrek-agent

.PHONY: clean
clean: ## Remove build artifacts.
	rm -rf bin/
	rm -f cover.out

##@ Docker

.PHONY: docker-build-agent
docker-build-agent: ## Build chrek-agent docker image.
	$(CONTAINER_TOOL) build --target agent -t ${IMG} .

.PHONY: docker-build-agent-lint
docker-build-agent-lint: ## Build chrek-agent docker image up to lint stage.
	$(CONTAINER_TOOL) build --target linter -t ${IMG}-lint .

.PHONY: docker-build-agent-test
docker-build-agent-test: ## Build chrek-agent docker image up to test stage.
	$(CONTAINER_TOOL) build --target tester -t ${IMG}-test .

.PHONY: docker-build-placeholder
docker-build-placeholder: ## Build placeholder image for checkpoint restore. Requires PLACEHOLDER_BASE_IMG.
ifndef PLACEHOLDER_BASE_IMG
	$(error PLACEHOLDER_BASE_IMG is required. Example: make docker-build-placeholder PLACEHOLDER_BASE_IMG=nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1-cuda13)
endif
	$(CONTAINER_TOOL) build --target placeholder \
		--build-arg BASE_IMAGE=${PLACEHOLDER_BASE_IMG} \
		-t ${PLACEHOLDER_IMG} .

.PHONY: docker-push-agent
docker-push-agent: ## Push chrek-agent docker image.
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
coverage: test ## Show test coverage report.
	go tool cover -func=cover.out
