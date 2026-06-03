# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared tool version pins and install rules.
# Include from root and per-component Makefiles AFTER defining REPO_ROOT.

LOCALBIN ?= $(REPO_ROOT)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

GOLANGCI_LINT_VERSION ?= v2.12.2
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint-$(GOLANGCI_LINT_VERSION)

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT)
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint,$(GOLANGCI_LINT_VERSION))

CONTROLLER_GEN_VERSION ?= v0.21.0
CONTROLLER_GEN = $(LOCALBIN)/controller-gen-$(CONTROLLER_GEN_VERSION)

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN)
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_GEN_VERSION))

ADDLICENSE_VERSION ?= v1.2.0
ADDLICENSE = $(LOCALBIN)/addlicense-$(ADDLICENSE_VERSION)

.PHONY: addlicense
addlicense: $(ADDLICENSE)
$(ADDLICENSE): $(LOCALBIN)
	$(call go-install-tool,$(ADDLICENSE),github.com/google/addlicense,$(ADDLICENSE_VERSION))

# go-install-tool installs a Go binary into $(LOCALBIN) under a versioned filename.
# Known fragility: the mv depends on a stable installed-binary filename. If a
# tool's installed name changes between major versions, the mv silently fails.
# Revisit when bumping pinned versions.
define go-install-tool
@[ -f $(1) ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv "$$(echo "$(1)" | sed "s/-$(3)$$//")" $(1) ;\
}
endef
