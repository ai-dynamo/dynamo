# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Module Makefile for operator/api. Self-contained: computes REPO_ROOT
# from its own location so it works under `make -C operator/api ...`
# regardless of the caller's CWD.

MODULE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT   := $(abspath $(MODULE_ROOT)/../..)

include $(REPO_ROOT)/hack/tools.mk

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

.PHONY: build
build: ## Compile the api module.
	go build ./...

.PHONY: tidy
tidy: ## go mod tidy.
	go mod tidy

.PHONY: generate
generate: $(CONTROLLER_GEN) ## Run controller-gen to produce deepcopy methods and CRD manifests.
	@CONTROLLER_GEN=$(CONTROLLER_GEN) $(MODULE_ROOT)/hack/generate.sh

.PHONY: lint
lint: $(GOLANGCI_LINT) ## Run golangci-lint.
	$(GOLANGCI_LINT) run

.PHONY: clean
clean: ## Remove generated artifacts.
	rm -f v1alpha1/zz_generated.deepcopy.go
	rm -rf v1alpha1/crds
