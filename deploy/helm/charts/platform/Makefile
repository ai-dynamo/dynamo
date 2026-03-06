# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

HELM_DOCS_VERSION ?= 1.14.2
HELM_DOCS ?= $(LOCALBIN)/helm-docs-$(HELM_DOCS_VERSION)

##@ General

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

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

.PHONY: generate-helm-docs
generate-helm-docs: helm-docs-install ## Generate README.md from values.yaml and README.md.gotmpl
	@echo "Generating Helm chart documentation..."
	@$(HELM_DOCS) \
		--template-files=README.md.gotmpl \
		--output-file=README.md \
		--sort-values-order=file \
		--chart-to-generate=. \
		--ignore-non-descriptions
	@rm -rf $(LOCALBIN)
	@echo "Generated README.md"

.PHONY: helm-docs-clean
helm-docs-clean: ## Remove generated helm documentation
	@rm -f README.md
	@echo "Cleaned generated README.md"