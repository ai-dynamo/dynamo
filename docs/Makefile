# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS        ?=
SPHINXBUILD       ?= sphinx-build
SOURCEDIR          = .
BUILDDIR           = build

##@ General

# Put it first so that "make" without argument is like "make help".
help: ## Display help for all targets
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional documentation targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "  \033[36m%-20s\033[0m %s\n", "TARGET", "DESCRIPTION"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

clean: ## Clean build artifacts
	@rm -fr ${BUILDDIR}

##@ Helm Documentation

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Versions
HELM_DOCS_VERSION ?= 1.14.2

## Tool Binaries
HELM_DOCS ?= $(LOCALBIN)/helm-docs-$(HELM_DOCS_VERSION)

.PHONY: helm-docs-install
helm-docs-install: $(HELM_DOCS) ## Download helm-docs locally if necessary
$(HELM_DOCS): $(LOCALBIN)
	@echo "📥 Downloading helm-docs $(HELM_DOCS_VERSION)..."
	@ARCH=$$(uname -m); \
	OS=$$(uname -s | tr '[:upper:]' '[:lower:]'); \
	curl -sSL "https://github.com/norwoodj/helm-docs/releases/download/v$(HELM_DOCS_VERSION)/helm-docs_$(HELM_DOCS_VERSION)_$${OS}_$${ARCH}.tar.gz" | \
	tar xz -C $(LOCALBIN) helm-docs && \
	mv $(LOCALBIN)/helm-docs $(HELM_DOCS) && \
	echo "✅ helm-docs $(HELM_DOCS_VERSION) installed successfully"

.PHONY: generate-helm-docs
generate-helm-docs: helm-docs-install ## Generate README.md for Helm charts from values.yaml
	@echo "📚 Generating Helm chart documentation..."
	@cd ../deploy/cloud/helm/platform && $(realpath $(HELM_DOCS)) \
		--template-files=README.md.gotmpl \
		--output-file=README.md \
		--sort-values-order=file \
		--chart-to-generate=. \
		--ignore-non-descriptions
	@echo "✅ Generated documentation at ../deploy/cloud/helm/platform/README.md"

.PHONY: helm-docs-clean
helm-docs-clean: ## Remove generated helm documentation
	@echo "🧹 Cleaning generated helm documentation..."
	@rm -f ../deploy/cloud/helm/platform/README.md
	@echo "✅ Cleaned helm documentation"

.PHONY: generate-crd-docs
generate-crd-docs: ## Generate CRD API reference documentation
	@echo "📚 Generating CRD API reference documentation..."
	@cd ../deploy/cloud/operator && make generate-api-docs
	@echo "✅ CRD API reference generated"

.PHONY: docs-all
docs-all: generate-helm-docs generate-crd-docs html ## Generate all documentation (Sphinx + Helm + CRDs)

.PHONY: help Makefile clean


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
