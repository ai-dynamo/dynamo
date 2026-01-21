
# Image URL to use all building/pushing image targets
IMG ?= controller:latest
# ENVTEST_K8S_VERSION refers to the version of kubebuilder assets to be downloaded by envtest binary.
ENVTEST_K8S_VERSION = 1.29.0

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# CONTAINER_TOOL defines the container tool to be used for building images.
# Be aware that the target commands are only tested with Docker which is
# scaffolded by default. However, you might want to replace it to use other
# tools. (i.e. podman)
CONTAINER_TOOL ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk command is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: check
check: generate manifests generate-api-docs
	@echo "> Checking for uncommitted changes"
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: Git tree is dirty after running validation steps."; \
		echo "Please check the diff to identify the step that dirtied the tree."; \
		git --no-pager status; \
		git --no-pager diff; \
		exit 1; \
	fi
	@echo "> Check complete"


.PHONY: ensure-yq
ensure-yq:
	@if ! command -v yq &> /dev/null; then \
		echo "Installing yq..."; \
		ARCH=$$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/'); \
		OS=$$(uname -s | tr '[:upper:]' '[:lower:]'); \
		wget https://github.com/mikefarah/yq/releases/latest/download/yq_$${OS}_$${ARCH} -O /usr/local/bin/yq && \
		chmod +x /usr/local/bin/yq; \
	else \
		echo "yq is already installed: $$(yq --version)"; \
	fi

.PHONY: manifests
manifests: controller-gen ensure-yq ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	# Use a large maxDescLen to ensure all field comments are included as OpenAPI descriptions
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd:maxDescLen=100000 webhook paths="./..." output:crd:artifacts:config=config/crd/bases
	echo "Removing name from mainContainer required fields"
	for file in config/crd/bases/*.yaml; do \
		yq eval '(.. | select(has("mainContainer")) | .mainContainer.required) |= (. - ["name"])' -i --indent 2 $$file || exit 1; \
	done
	echo "Removing containers from extraPodSpec required fields"
	for file in config/crd/bases/*.yaml; do \
		yq eval '(.. | select(has("extraPodSpec")) | .extraPodSpec.required) |= (. - ["containers"])' -i --indent 2 $$file || exit 1; \
	done
	echo "Adding NVIDIA header to CRD files"
	for file in config/crd/bases/*.yaml; do \
		if ! head -20 "$$file" | grep -q "NVIDIA CORPORATION"; then \
			{ printf '%s\n' \
				'# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.' \
				'# SPDX-License-Identifier: Apache-2.0' \
				'#' \
				'# Licensed under the Apache License, Version 2.0 (the "License");' \
				'# you may not use this file except in compliance with the License.' \
				'# You may obtain a copy of the License at' \
				'#' \
				'# http://www.apache.org/licenses/LICENSE-2.0' \
				'#' \
				'# Unless required by applicable law or agreed to in writing, software' \
				'# distributed under the License is distributed on an "AS IS" BASIS,' \
				'# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.' \
				'# See the License for the specific language governing permissions and' \
				'# limitations under the License.' \
				''; \
			  cat "$$file"; \
			} > "$$file.tmp" && mv "$$file.tmp" "$$file"; \
		fi; \
	done
	echo "Adding helm.sh/resource-policy: keep annotation to CRD files"
	for file in config/crd/bases/*.yaml; do \
		if ! yq eval '.metadata.annotations."helm.sh/resource-policy"' "$$file" | grep -q "keep"; then \
			yq eval '.metadata.annotations."helm.sh/resource-policy" = "keep"' -i "$$file"; \
		fi; \
	done
	if [ ! -d "../helm/charts/crds/templates/" ]; then \
		echo "Creating directory ../helm/charts/crds/templates/"; \
		mkdir -p ../helm/charts/crds/templates/; \
	fi
	cp config/crd/bases/*.yaml ../helm/charts/crds/templates/
	echo "Adding NVIDIA header to RBAC files"
	for file in config/rbac/*.yaml; do \
		if [ -f "$$file" ] && ! head -20 "$$file" | grep -q "NVIDIA CORPORATION"; then \
			{ printf '%s\n' \
				'# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.' \
				'# SPDX-License-Identifier: Apache-2.0' \
				'#' \
				'# Licensed under the Apache License, Version 2.0 (the "License");' \
				'# you may not use this file except in compliance with the License.' \
				'# You may obtain a copy of the License at' \
				'#' \
				'# http://www.apache.org/licenses/LICENSE-2.0' \
				'#' \
				'# Unless required by applicable law or agreed to in writing, software' \
				'# distributed under the License is distributed on an "AS IS" BASIS,' \
				'# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.' \
				'# See the License for the specific language governing permissions and' \
				'# limitations under the License.' \
				''; \
			  cat "$$file"; \
			} > "$$file.tmp" && mv "$$file.tmp" "$$file"; \
		fi; \
	done

.PHONY: generate
generate: controller-gen ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt_" paths="./..."

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: test
test: manifests generate fmt vet envtest ## Run tests.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test $$(go list ./... | grep -v /e2e | grep -v /test | grep -v /api | grep -v /cmd) -coverprofile cover.out

# Utilize Kind or modify the e2e tests to load the image locally, enabling compatibility with other vendors.
.PHONY: test-e2e  # Run the e2e tests against a Kind k8s instance that is spun up.
test-e2e:
	go test ./test/e2e/ -v -ginkgo.v

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter & yamllint
	$(GOLANGCI_LINT) run

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes
	$(GOLANGCI_LINT) run --fix

##@ Build

.PHONY: build
build: manifests generate fmt vet helm ## Build manager binary.
	go build -o bin/manager cmd/main.go

.PHONY: run
run: manifests generate fmt vet ## Run a controller from your host.
	go run ./cmd/main.go

# If you wish to build the manager image targeting other platforms you can use the --platform flag.
# (i.e. docker build --platform linux/arm64). However, you must enable docker buildKit for it.
# More info: https://docs.docker.com/develop/develop-images/build_enhancements/
.PHONY: docker-build
docker-build: ## Build docker image with the manager.
	$(CONTAINER_TOOL) build -t ${IMG} .

.PHONY: docker-push
docker-push: ## Push docker image with the manager.
	$(CONTAINER_TOOL) push ${IMG}

# PLATFORMS defines the target platforms for the manager image be built to provide support to multiple
# architectures. (i.e. make docker-buildx IMG=myregistry/mypoperator:0.0.1). To use this option you need to:
# - be able to use docker buildx. More info: https://docs.docker.com/build/buildx/
# - have enabled BuildKit. More info: https://docs.docker.com/develop/develop-images/build_enhancements/
# - be able to push the image to your registry (i.e. if you do not set a valid value via IMG=<myregistry/image:<tag>> then the export will fail)
# To adequately provide solutions that are compatible with multiple platforms, you should consider using this option.
PLATFORMS ?= linux/arm64,linux/amd64,linux/s390x,linux/ppc64le
.PHONY: docker-buildx
docker-buildx: ## Build and push docker image for the manager for cross-platform support
	# copy existing Dockerfile and insert --platform=${BUILDPLATFORM} into Dockerfile.cross, and preserve the original Dockerfile
	sed -e '1 s/\(^FROM\)/FROM --platform=\$$\{BUILDPLATFORM\}/; t' -e ' 1,// s//FROM --platform=\$$\{BUILDPLATFORM\}/' Dockerfile > Dockerfile.cross
	- $(CONTAINER_TOOL) buildx create --name project-v3-builder
	$(CONTAINER_TOOL) buildx use project-v3-builder
	- $(CONTAINER_TOOL) buildx build --push --platform=$(PLATFORMS) --tag ${IMG} -f Dockerfile.cross .
	- $(CONTAINER_TOOL) buildx rm project-v3-builder
	rm Dockerfile.cross

.PHONY: build-installer
build-installer: manifests generate kustomize ## Generate a consolidated YAML with CRDs and deployment.
	mkdir -p dist
	@if [ -d "config/crd" ]; then \
		$(KUSTOMIZE) build config/crd > dist/install.yaml; \
	fi
	echo "---" >> dist/install.yaml  # Add a document separator before appending
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default >> dist/install.yaml

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: manifests kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) apply -f -

.PHONY: uninstall
uninstall: manifests kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) delete --ignore-not-found=$(ignore-not-found) -f -

.PHONY: deploy
deploy: manifests kustomize ## Deploy controller to the K8s cluster specified in ~/.kube/config.
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default | $(KUBECTL) apply -f -

.PHONY: undeploy
undeploy: kustomize ## Undeploy controller from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/default | $(KUBECTL) delete --ignore-not-found=$(ignore-not-found) -f -

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
KUBECTL ?= kubectl
KUSTOMIZE ?= $(LOCALBIN)/kustomize-$(KUSTOMIZE_VERSION)
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen-$(CONTROLLER_TOOLS_VERSION)
ENVTEST ?= $(LOCALBIN)/setup-envtest-$(ENVTEST_VERSION)
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint-$(GOLANGCI_LINT_VERSION)

## Tool Versions
KUSTOMIZE_VERSION ?= v5.5.0
CONTROLLER_TOOLS_VERSION ?= v0.16.4
ENVTEST_VERSION ?= release-0.19
GOLANGCI_LINT_VERSION ?= v1.64.8

.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	$(call go-install-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v5,$(KUSTOMIZE_VERSION))

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: envtest
envtest: $(ENVTEST) ## Download setup-envtest locally if necessary.
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,$(ENVTEST_VERSION))

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

######################### Helmify
HELMIFY ?= $(LOCALBIN)/helmify

.PHONY: helmify
helmify: $(HELMIFY) ## Download helmify locally if necessary.
$(HELMIFY): $(LOCALBIN)
	test -s $(LOCALBIN)/helmify || GOBIN=$(LOCALBIN) go install github.com/arttor/helmify/cmd/helmify@v0.4.16

helm: manifests kustomize helmify
	$(KUSTOMIZE) build config/default | $(HELMIFY) -image-pull-secrets charts/dynamo-kubernetes-operator

######################### CRD Reference Docs
CRD_REF_DOCS_VERSION ?= latest
CRD_REF_DOCS ?= $(LOCALBIN)/crd-ref-docs

.PHONY: crd-ref-docs
crd-ref-docs: $(CRD_REF_DOCS) ## Download crd-ref-docs locally if necessary.
$(CRD_REF_DOCS): $(LOCALBIN)
	@echo "Installing crd-ref-docs $(CRD_REF_DOCS_VERSION)"
	@GOBIN=$(LOCALBIN) go install github.com/elastic/crd-ref-docs@$(CRD_REF_DOCS_VERSION)
	@echo "✅ crd-ref-docs $(CRD_REF_DOCS_VERSION) installed successfully"

.PHONY: generate-api-docs
generate-api-docs: crd-ref-docs ## Generate API reference documentation from CRDs
	@echo "📚 Generating CRD API reference documentation..."
	@mkdir -p docs
	@$(CRD_REF_DOCS) \
		--source-path=api \
		--config=docs/crd-ref-docs-config.yaml \
		--renderer=markdown \
		--output-path=./docs/api_reference.md
	@echo "✅ Generated API reference at ./docs/api_reference.md"
	# concatenate header.md, api_reference.md, and footer.md
	cat docs/header.md ./docs/api_reference.md docs/footer.md > ../../docs/kubernetes/api_reference.md
	rm ./docs/api_reference.md
	@echo "✅ Concatenated header.md, api_reference.md, and footer.md"

.PHONY: coverage
coverage: test
	go tool cover -func=cover.out
