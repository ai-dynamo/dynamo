# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

HELM_UNITTEST_VERSION ?= v1.0.3

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
lint: ## Run helm lint and render production/dev paths
	@helm lint . --set image.tag=ci-test
	@helm template power-agent . --set image.tag=ci-test >/dev/null
	@helm template power-agent . \
		--set daemonset.enabled=false \
		--set dev.enabled=true \
		--set dev.nodeName=ci-node >/dev/null

.PHONY: test
test: helm-unittest-install ## Run helm-unittest chart tests
	@helm unittest .
