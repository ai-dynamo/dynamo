# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Root orchestrator Makefile. Delegates component work to per-component Makefiles
# (agent/Makefile). Module-wide targets (fmt/vet/test/lint/tidy)
# stay here because snapshot is currently a single Go module.

REPO_ROOT := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
VERSION   := $(shell cat "$(REPO_ROOT)/VERSION")

include hack/tools.mk

# Image tag exported to component docker targets (overridable from the env).
IMG ?= ghcr.io/run-ai/snapshot/agent:$(VERSION)
export IMG

# Containerized-test settings (used by test-qemu). CONTAINER_TOOL mirrors the var in
# agent/Makefile — keep the two in sync.
CONTAINER_TOOL ?= docker
TEST_IMAGE     ?= golang:1.26
# Default to linux/amd64 (the production target), emulated via QEMU on Apple Silicon.
# Override for a faster native run on Apple Silicon: make test-qemu TEST_PLATFORM=linux/arm64
TEST_PLATFORM  ?= linux/amd64
GOCACHE_DIR    ?= /cache/go-build

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.DEFAULT_GOAL := help

.PHONY: all
all: build

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Module-wide
# Today the agent owns the root go.mod. When operator's own go.mod lands,
# add per-module delegation here (e.g., $(MAKE) -C operator test).

.PHONY: fmt
fmt: ## Run go fmt across the root module.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet across the root module.
	go vet ./...

.PHONY: test
test: ## Run unit tests for the root module.
	go test ./... -coverprofile cover.out

.PHONY: test-qemu
test-qemu: ## Run unit tests in a linux/amd64 container (for non-Linux hosts, e.g. macOS via QEMU; TEST_PLATFORM overrides arch).
	@command -v $(CONTAINER_TOOL) >/dev/null 2>&1 || { \
		echo >&2 "$(CONTAINER_TOOL) not found — install Docker or set CONTAINER_TOOL=podman"; exit 1; }
	$(CONTAINER_TOOL) run --rm \
		--platform $(TEST_PLATFORM) \
		-v "$(REPO_ROOT)":/workspace -w /workspace \
		-v snapshot-go-mod:/go/pkg/mod \
		-v snapshot-go-build:$(GOCACHE_DIR) \
		-e GOCACHE=$(GOCACHE_DIR) \
		-e GOFLAGS=-buildvcs=false \
		$(TEST_IMAGE) \
		go test ./...

.PHONY: coverage
coverage: test ## Print coverage by function.
	go tool cover -func=cover.out

.PHONY: lint
lint: golangci-lint ## Run golangci-lint.
	$(GOLANGCI_LINT) run

.PHONY: tidy
tidy: ## go mod tidy on the root module.
	go mod tidy

##@ API / Codegen

.PHONY: generate
generate: ## Generate deepcopy + CRD manifests via operator/api.
	$(MAKE) --directory=operator/api generate

.PHONY: manifests
manifests: generate ## Alias for generate (kubebuilder convention).

.PHONY: verify-generate
verify-generate: generate ## Fail if generated artifacts have drifted from Go types.
	@git diff --exit-code -- operator/api/ || { \
		echo >&2 "operator/api/ has uncommitted changes after 'make generate'"; \
		echo >&2 "regenerate locally and commit the diff"; \
		exit 1; \
	}

.PHONY: build-api
build-api: ## Compile operator/api.
	$(MAKE) --directory=operator/api build

.PHONY: check
check: verify-generate verify-license-headers fmt vet lint ## Aggregate verification: drift gate + license + fmt/vet/lint.

##@ Build (delegates to per-component)

.PHONY: build
build: build-agent build-api ## Build all component binaries.

.PHONY: build-agent
build-agent: ## Build agent binaries via agent/Makefile.
	$(MAKE) --directory=agent build

##@ Container (agent only)

.PHONY: docker-build
docker-build: ## Build agent container image via agent/Makefile.
	$(MAKE) --directory=agent docker-build

.PHONY: docker-push
docker-push: ## Push agent container image. Retry standalone on deploy failure.
	$(MAKE) --directory=agent docker-push

.PHONY: deploy
deploy: ## docker-build + docker-push (agent).
	$(MAKE) --directory=agent deploy

##@ License

.PHONY: add-license-headers
add-license-headers: addlicense ## Add SPDX license headers to source files (idempotent).
	ADDLICENSE=$(ADDLICENSE) hack/add-license-headers.sh

.PHONY: verify-license-headers
verify-license-headers: addlicense ## Fail if any source file is missing an SPDX license header.
	ADDLICENSE=$(ADDLICENSE) hack/add-license-headers.sh --check

##@ Cleanup

.PHONY: clean
clean: ## Remove build artifacts.
	$(MAKE) --directory=agent clean
	rm -rf $(REPO_ROOT)/bin cover.out
