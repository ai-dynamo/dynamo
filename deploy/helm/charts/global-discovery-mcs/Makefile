# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

HELM_UNITTEST_VERSION ?= v1.0.3

LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

HELM_DOCS_VERSION ?= 1.14.2
HELM_DOCS ?= $(LOCALBIN)/helm-docs-$(HELM_DOCS_VERSION)

# The module whose image, CRDs and controller ClusterRole this chart deploys.
MODULE_DIR ?= ../../../global-discovery-mcs
MODULE_CRDS ?= $(MODULE_DIR)/config/crd/bases
CHART_CRDS ?= files/crds

# controller.enabled (the default) requires a registry URL, so every render
# sets one.
LINT_VALUES = --set controller.registryUrls[0]=https://mcs.example.internal

# Values that render every template: the registry with the DynamoDB store,
# TLS and its optional objects, and the controller with a CA bundle and its
# optional objects.
LINT_ALL_VALUES = \
	--set registry.enabled=true \
	--set registry.store=dynamodb \
	--set registry.replicas=2 \
	--set registry.tls.secretName=mcs-registry-tls \
	--set registry.podDisruptionBudget.enabled=true \
	--set registry.metrics.serviceMonitor.enabled=true \
	--set controller.registryTlsCa.configMapName=mcs-registry-ca \
	--set controller.metrics.serviceMonitor.enabled=true

##@ General

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Helm Testing

.PHONY: helm-unittest-install
helm-unittest-install: ## Install helm-unittest plugin if not present
	@helm plugin list | grep -q unittest || \
		helm plugin install https://github.com/helm-unittest/helm-unittest --version $(HELM_UNITTEST_VERSION)

.PHONY: lint
lint: verify-crds ## Run helm lint and render the default, full and CRD-less paths
	@helm lint --strict . $(LINT_VALUES)
	@helm lint --strict . $(LINT_VALUES) $(LINT_ALL_VALUES)
	@helm template global-discovery-mcs . $(LINT_VALUES) >/dev/null
	@helm template global-discovery-mcs . $(LINT_VALUES) $(LINT_ALL_VALUES) >/dev/null
	@helm template global-discovery-mcs . \
		--set crds.install=false \
		--set controller.enabled=false \
		--set registry.enabled=true >/dev/null

.PHONY: test
test: helm-unittest-install ## Run helm-unittest chart tests
	@helm unittest .

##@ CRDs

.PHONY: sync-crds
sync-crds: ## Copy the CRDs of deploy/global-discovery-mcs/config/crd/bases into files/crds
	@mkdir -p $(CHART_CRDS)
	@rm -f $(CHART_CRDS)/*.yaml
	@cp $(MODULE_CRDS)/*.yaml $(CHART_CRDS)/
	@echo "$(CHART_CRDS) holds the CRDs of $(MODULE_CRDS)"

.PHONY: verify-crds
verify-crds: ## Fail when files/crds differs from deploy/global-discovery-mcs/config/crd/bases
	@if ! diff -r $(MODULE_CRDS) $(CHART_CRDS); then \
		echo "ERROR: $(CHART_CRDS) differs from $(MODULE_CRDS); run make sync-crds" >&2; \
		exit 1; \
	fi

##@ Helm Documentation

.PHONY: helm-docs-install
helm-docs-install: $(HELM_DOCS) ## Download helm-docs locally if necessary
$(HELM_DOCS): $(LOCALBIN)
	@echo "Downloading helm-docs $(HELM_DOCS_VERSION)..."
	@ARCH=$$(uname -m); \
	case $$ARCH in aarch64) ARCH=arm64;; esac; \
	OS=$$(uname -s); \
	curl -fsSL "https://github.com/norwoodj/helm-docs/releases/download/v$(HELM_DOCS_VERSION)/helm-docs_$(HELM_DOCS_VERSION)_$${OS}_$${ARCH}.tar.gz" | \
	tar xz -C $(LOCALBIN) helm-docs && \
	mv $(LOCALBIN)/helm-docs $(HELM_DOCS)

.PHONY: docs
docs: helm-docs-install ## Generate README.md from values.yaml and README.md.gotmpl
	@echo "Generating Helm chart documentation..."
	@$(HELM_DOCS) \
		--template-files=README.md.gotmpl \
		--output-file=README.md \
		--sort-values-order=file \
		--chart-to-generate=. \
		--ignore-non-descriptions
	@rm -rf $(LOCALBIN)
	@echo "Generated README.md"
