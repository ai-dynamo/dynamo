# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODULE_ROOT := $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))")
REPO_ROOT   := $(shell dirname "$(MODULE_ROOT)")
VERSION     := $(shell cat "$(REPO_ROOT)/VERSION")

include $(REPO_ROOT)/hack/tools.mk

IMG ?= nvcr.io/nvidian/dynamo-dev/nvsnapshot-agent:$(VERSION)
CONTAINER_TOOL ?= docker
RUNTIME_IMAGE_PLATFORM ?= linux/amd64
CRIU_BUILD_ARGS = $(if $(CRIU_REPO),--build-arg CRIU_REPO=$(CRIU_REPO),) $(if $(CRIU_REF),--build-arg CRIU_REF=$(CRIU_REF),)

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make -C agent \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

.PHONY: build
build: $(LOCALBIN) ## Cross-compile snapshot-agent and nsrestore to bin/ (linux/amd64).
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -C $(REPO_ROOT) -o $(LOCALBIN)/snapshot-agent ./agent/cmd/agent
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -C $(REPO_ROOT) -o $(LOCALBIN)/nsrestore     ./agent/cmd/nsrestore

.PHONY: clean
clean: ## Remove agent binaries from bin/.
	rm -f $(LOCALBIN)/snapshot-agent $(LOCALBIN)/nsrestore

##@ Container

.PHONY: docker-build
docker-build: ## Build the snapshot-agent container image.
	$(CONTAINER_TOOL) build --platform $(RUNTIME_IMAGE_PLATFORM) --target agent \
		$(CRIU_BUILD_ARGS) \
		-f $(MODULE_ROOT)/Dockerfile \
		-t $(IMG) $(REPO_ROOT)

.PHONY: docker-push
docker-push: ## Push the snapshot-agent container image. Retry standalone on deploy failure.
	$(CONTAINER_TOOL) push $(IMG)

.PHONY: deploy
deploy: docker-build docker-push ## docker-build then docker-push.
