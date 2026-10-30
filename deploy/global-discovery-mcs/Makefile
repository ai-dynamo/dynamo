# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Multi-Cluster Services registry (mcs-registry), its Kubernetes
# controller (mcs-controller) and admin CLI (mcs-admin).

MODULE := github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs

# Resolve against this module's go.mod even when a go.work exists above it.
export GOWORK := off
GO ?= go
# The Go version of go.mod; tools that read the module are built with it.
GO_VERSION := $(shell sed -n 's/^go //p' go.mod)

# Image built by docker-build. It holds the three binaries and the CRDs.
IMG ?= global-discovery-mcs:latest
CONTAINER_TOOL ?= docker

# Build metadata linked into the binaries.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
GIT_HASH ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
LDFLAGS ?= -s -w -X $(MODULE)/internal/version.Version=$(VERSION) -X $(MODULE)/internal/version.GitHash=$(GIT_HASH)

# Minimum statement coverage of ./internal/..., in percent.
COVERAGE_MIN ?= 85

# ENVTEST_K8S_VERSION is the kube-apiserver and etcd release of the envtest
# suite, the same as deploy/operator.
ENVTEST_K8S_VERSION = 1.30.0
ENVTEST_PACKAGES ?= ./internal/mcs/integration/...
ENVTEST_ASSETS_CMD = $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path

# Integration tests (build tag integration) run the DynamoDB store and
# mcs-admin create-table against amazon/dynamodb-local in Docker.
INTEGRATION_PACKAGES ?= ./internal/store/dynamodb/... ./cmd/mcs-admin/...

# The controller ClusterRole is generated into config/rbac and copied into
# the Helm chart.
CHART_DIR ?= ../helm/charts/global-discovery-mcs
CHART_ROLE ?= $(CHART_DIR)/files/controller-role.yaml
# The chart renders a copy of config/crd/bases; make -C $(CHART_DIR) sync-crds
# refreshes it.
CHART_CRDS ?= $(CHART_DIR)/files/crds

# CRDs of sigs.k8s.io/mcs-api that sync-crds copies into config/crd/bases.
MCS_API_CRDS = multicluster.x-k8s.io_serviceexports.yaml multicluster.x-k8s.io_serviceimports.yaml

GEN_DIR := api/gen
SPDX_COPYRIGHT := SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX_LICENSE := SPDX-License-Identifier: Apache-2.0

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'.

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Run go fmt against code.
	$(GO) fmt ./...

.PHONY: vet
vet: ## Run go vet against code, integration tests included.
	$(GO) vet ./...
	$(GO) vet -tags integration ./...

.PHONY: lint
lint: golangci-lint ## Run golangci-lint, integration tests included.
	$(GOLANGCI_LINT) run --build-tags integration

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint and apply its fixes.
	$(GOLANGCI_LINT) run --build-tags integration --fix

.PHONY: test
test: fmt vet setup-envtest ## Run the unit and envtest suites with the race detector and the coverage gate.
	KUBEBUILDER_ASSETS="$$($(ENVTEST_ASSETS_CMD))" $(GO) test -race -covermode=atomic -coverpkg=./internal/... -coverprofile=cover.out ./...
	@total="$$($(GO) tool cover -func=cover.out | tail -1)"; \
	echo "$${total}"; \
	pct="$$(awk '{ sub(/%/, "", $$NF); print $$NF }' <<<"$${total}")"; \
	if awk -v p="$${pct}" -v m="$(COVERAGE_MIN)" 'BEGIN { exit !(p < m) }'; then \
		echo "coverage of ./internal/... is $${pct}%, below $(COVERAGE_MIN)%" >&2; exit 1; \
	fi

.PHONY: envtest
envtest: setup-envtest ## Run the mcs-controller envtest suite against kube-apiserver and etcd.
	KUBEBUILDER_ASSETS="$$($(ENVTEST_ASSETS_CMD))" $(GO) test -race -count=1 $(ENVTEST_PACKAGES) -v

# DOCKER_HOST defaults to the endpoint of the current docker context, and
# colima gets the Ryuk socket override it needs. MCS_TEST_DYNAMODB_ENDPOINT
# (for example http://localhost:8000) uses a running endpoint instead.
.PHONY: test-integration
test-integration: ## Run the DynamoDB store and mcs-admin integration tests against dynamodb-local (Docker).
	@docker_host="$${DOCKER_HOST:-$$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null)}"; \
	socket_override="$${TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE:-}"; \
	case "$${docker_host}" in *colima*) socket_override="$${socket_override:-/var/run/docker.sock}" ;; esac; \
	echo "DOCKER_HOST=$${docker_host} $(GO) test -race -count=1 -tags=integration $(INTEGRATION_PACKAGES)"; \
	DOCKER_HOST="$${docker_host}" TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE="$${socket_override}" \
		$(GO) test -race -count=1 -tags=integration $(INTEGRATION_PACKAGES)

.PHONY: manifests
manifests: controller-gen ## Generate the controller ClusterRole from the +kubebuilder:rbac markers.
	$(CONTROLLER_GEN) rbac:roleName=mcs-controller,headerFile=hack/boilerplate.yaml.txt paths="./internal/mcs/..." output:rbac:dir=config/rbac
	mkdir -p $(dir $(CHART_ROLE))
	cp config/rbac/role.yaml $(CHART_ROLE)

# add-go-headers adds the license header to the grpc-gateway files under $(1),
# since protoc-gen-grpc-gateway does not copy it from the proto sources.
define add-go-headers
for f in $$(find $(1) -type f -name '*.pb.gw.go'); do \
	if ! head -n 3 "$$f" | grep -q 'SPDX-License-Identifier'; then \
		{ printf '%s\n' '// $(SPDX_COPYRIGHT)' '// $(SPDX_LICENSE)' ''; cat "$$f"; } > "$$f.tmp" && mv "$$f.tmp" "$$f"; \
	fi; \
done
endef

.PHONY: proto
proto: proto-tools ## Lint the proto contract and regenerate api/gen.
	$(BUF) lint
	mkdir -p $(GEN_DIR)
	find $(GEN_DIR) -type f \( -name '*.pb.go' -o -name '*.pb.gw.go' -o -name '*.swagger.json' \) -delete
	$(BUF) generate --template buf.gen.yaml
	@$(call add-go-headers,$(GEN_DIR))

.PHONY: proto-check
proto-check: proto-tools ## Lint the proto contract and fail when api/gen differs from it.
	$(BUF) lint
	@tmp="$$(mktemp -d)"; trap 'rm -rf "$${tmp}"' EXIT; \
	$(BUF) generate --template buf.gen.yaml -o "$${tmp}"; \
	$(call add-go-headers,$${tmp}/$(GEN_DIR)); \
	if ! diff -ru $(GEN_DIR) "$${tmp}/$(GEN_DIR)"; then \
		echo "error: $(GEN_DIR) is stale; run make proto and commit the result" >&2; exit 1; \
	fi; \
	echo "$(GEN_DIR) is up to date"

.PHONY: sync-crds
sync-crds: ## Copy the ServiceExport and ServiceImport CRDs of sigs.k8s.io/mcs-api into config/crd/bases.
	$(GO) mod download sigs.k8s.io/mcs-api
	@dir="$$($(GO) list -m -f '{{.Dir}}' sigs.k8s.io/mcs-api)"; \
	version="$$($(GO) list -m -f '{{.Version}}' sigs.k8s.io/mcs-api)"; \
	for f in $(MCS_API_CRDS); do \
		{ printf '%s\n' '# $(SPDX_COPYRIGHT)' '# $(SPDX_LICENSE)' '#' \
			"# Copied unchanged from sigs.k8s.io/mcs-api@$${version}/config/crd/$${f}" \
			'# by make sync-crds; the upstream header follows.' '#'; \
		  cat "$${dir}/config/crd/$${f}"; } > "config/crd/bases/$${f}"; \
	done; \
	echo "config/crd/bases holds the CRDs of sigs.k8s.io/mcs-api $${version}"

.PHONY: tidy-check
tidy-check: ## Fail when go mod tidy would change go.mod or go.sum.
	@$(GO) mod tidy -diff || { echo "error: go.mod or go.sum is not tidy; run go mod tidy and commit the result" >&2; exit 1; }

.PHONY: check-chart-crds
check-chart-crds: ## Fail when the chart's copy of the CRDs differs from config/crd/bases.
	@if ! diff -r config/crd/bases $(CHART_CRDS); then \
		echo "ERROR: $(CHART_CRDS) differs from config/crd/bases; run make -C $(CHART_DIR) sync-crds" >&2; exit 1; \
	fi

.PHONY: check
check: tidy-check manifests proto sync-crds check-chart-crds ## Check go mod tidy, regenerate every generated file and fail when the tree changed.
	@echo "> Checking for uncommitted changes"
	@if [ -n "$$(git status --porcelain -- . $(CHART_DIR))" ]; then \
		echo "ERROR: the tree is dirty after regenerating; commit the generated files."; \
		git --no-pager status -- . $(CHART_DIR); \
		git --no-pager diff -- . $(CHART_DIR); \
		exit 1; \
	fi
	@echo "> Check complete"

##@ Build

.PHONY: build
build: fmt vet ## Build mcs-registry, mcs-controller and mcs-admin into bin/.
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(LOCALBIN)/mcs-registry ./cmd/mcs-registry
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(LOCALBIN)/mcs-controller ./cmd/mcs-controller
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(LOCALBIN)/mcs-admin ./cmd/mcs-admin

.PHONY: docker-build
docker-build: ## Build the image with the three binaries.
	$(CONTAINER_TOOL) build -t $(IMG) .

.PHONY: docker-push
docker-push: ## Push the image.
	$(CONTAINER_TOOL) push $(IMG)

##@ End-to-end

# The end-to-end test (test/e2e) builds the image, creates three kind clusters
# (hub, a and b), installs the chart and runs the phases; see test/e2e/README.md.
# Settings are environment variables or make variables: E2E_PHASES, E2E_KEEP
# (1 keeps the clusters when the run fails, always also after a pass),
# E2E_SKIP_BUILD (1 reuses the images of an earlier run), E2E_TIMEOUT.
E2E_DIR ?= test/e2e
E2E_PROVIDER ?= dynamo
E2E_PHASES ?= $(if $(filter cilium,$(E2E_PROVIDER)),3,1 2 3)
export E2E_PROVIDER E2E_PHASES

.PHONY: e2e
e2e: kind ## Run the kind end-to-end test of the registry, the controller and the Global Discovery Service.
	KIND=$(KIND) $(E2E_DIR)/run.sh

.PHONY: e2e-runner-test
e2e-runner-test: ## Check runner cleanup and exit statuses without creating clusters.
	bash $(E2E_DIR)/runner_test.sh

.PHONY: e2e-discovery
e2e-discovery: e2e-runner-test kind ## Run the same strict discovery cases with E2E_PROVIDER=dynamo or cilium.
	E2E_PHASES=3 E2E_STRICT=1 KIND=$(KIND) $(E2E_DIR)/run.sh

.PHONY: e2e-down
e2e-down: kind ## Delete the kind clusters of the end-to-end test.
	KIND=$(KIND) $(E2E_DIR)/run.sh down

##@ Upstream MCS conformance

# The opt-in build tag keeps real-cluster tests out of the unit/envtest run.
.PHONY: conformance-bin
conformance-bin: $(LOCALBIN) ## Compile the upstream MCS conformance runner.
	$(GO) test -tags conformance -c -o $(LOCALBIN)/mcs-conformance.test ./test/conformance

.PHONY: conformance
conformance: kind conformance-bin ## Run the full upstream suite on two isolated kind clusters.
	KIND=$(KIND) test/conformance/run.sh

.PHONY: conformance-setup
conformance-setup: kind ## Create and deploy the isolated conformance clusters, retaining them.
	KIND=$(KIND) test/conformance/run.sh setup

.PHONY: conformance-test
conformance-test: kind conformance-bin ## Run upstream tests against retained conformance clusters.
	KIND=$(KIND) test/conformance/run.sh test

.PHONY: conformance-down
conformance-down: kind ## Delete only the isolated conformance clusters.
	KIND=$(KIND) test/conformance/run.sh down

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(CURDIR)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

.PHONY: clean-bin
clean-bin: ## Remove the built binaries and the downloaded tools.
	rm -rf $(LOCALBIN)

## Prefer project-local binaries over system-wide ones.
export PATH := $(LOCALBIN):$(PATH)

## Tool Versions
CONTROLLER_TOOLS_VERSION ?= v0.21.0
ENVTEST_VERSION ?= release-0.24
GOLANGCI_LINT_VERSION ?= v2.15.0
BUF_VERSION ?= v1.75.0
PROTOC_GEN_GO_VERSION ?= v1.36.12
PROTOC_GEN_GO_GRPC_VERSION ?= v1.6.2
GRPC_GATEWAY_VERSION ?= v2.29.0
KIND_VERSION ?= v0.33.0

## Tool Binaries
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen-$(CONTROLLER_TOOLS_VERSION)
ENVTEST ?= $(LOCALBIN)/setup-envtest-$(ENVTEST_VERSION)
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint-$(GOLANGCI_LINT_VERSION)
BUF ?= $(LOCALBIN)/buf-$(BUF_VERSION)
PROTOC_GEN_GO = $(LOCALBIN)/protoc-gen-go-$(PROTOC_GEN_GO_VERSION)
PROTOC_GEN_GO_GRPC = $(LOCALBIN)/protoc-gen-go-grpc-$(PROTOC_GEN_GO_GRPC_VERSION)
PROTOC_GEN_GRPC_GATEWAY = $(LOCALBIN)/protoc-gen-grpc-gateway-$(GRPC_GATEWAY_VERSION)
PROTOC_GEN_OPENAPIV2 = $(LOCALBIN)/protoc-gen-openapiv2-$(GRPC_GATEWAY_VERSION)
KIND ?= $(LOCALBIN)/kind-$(KIND_VERSION)

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: setup-envtest
setup-envtest: $(ENVTEST) ## Download setup-envtest and the envtest Kubernetes assets locally if necessary.
	$(ENVTEST_ASSETS_CMD) >/dev/null
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,$(ENVTEST_VERSION))

# golangci-lint must be built with at least the Go version of go.mod to load
# the module.
.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint,$(GOLANGCI_LINT_VERSION),go$(GO_VERSION)+auto)

.PHONY: kind
kind: $(KIND) ## Download kind locally if necessary.
$(KIND): $(LOCALBIN)
	$(call go-install-tool,$(KIND),sigs.k8s.io/kind,$(KIND_VERSION))

# buf finds the protoc plugins on PATH under their plain names, so each
# plugin also gets an unversioned link in LOCALBIN.
.PHONY: proto-tools
proto-tools: $(BUF) $(PROTOC_GEN_GO) $(PROTOC_GEN_GO_GRPC) $(PROTOC_GEN_GRPC_GATEWAY) $(PROTOC_GEN_OPENAPIV2) ## Download buf and the protoc plugins locally if necessary.
	ln -sf $(notdir $(PROTOC_GEN_GO)) $(LOCALBIN)/protoc-gen-go
	ln -sf $(notdir $(PROTOC_GEN_GO_GRPC)) $(LOCALBIN)/protoc-gen-go-grpc
	ln -sf $(notdir $(PROTOC_GEN_GRPC_GATEWAY)) $(LOCALBIN)/protoc-gen-grpc-gateway
	ln -sf $(notdir $(PROTOC_GEN_OPENAPIV2)) $(LOCALBIN)/protoc-gen-openapiv2
$(BUF): $(LOCALBIN)
	$(call go-install-tool,$(BUF),github.com/bufbuild/buf/cmd/buf,$(BUF_VERSION))
$(PROTOC_GEN_GO): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_GO),google.golang.org/protobuf/cmd/protoc-gen-go,$(PROTOC_GEN_GO_VERSION))
$(PROTOC_GEN_GO_GRPC): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_GO_GRPC),google.golang.org/grpc/cmd/protoc-gen-go-grpc,$(PROTOC_GEN_GO_GRPC_VERSION))
$(PROTOC_GEN_GRPC_GATEWAY): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_GRPC_GATEWAY),github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway,$(GRPC_GATEWAY_VERSION))
$(PROTOC_GEN_OPENAPIV2): $(LOCALBIN)
	$(call go-install-tool,$(PROTOC_GEN_OPENAPIV2),github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2,$(GRPC_GATEWAY_VERSION))

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary (ideally with version)
# $2 - package url which can be installed
# $3 - specific version of package
# $4 - optional GOTOOLCHAIN for the build
define go-install-tool
@[ -f $(1) ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
GOBIN=$(LOCALBIN) $(if $(4),GOTOOLCHAIN=$(4)) $(GO) install $${package} ;\
mv "$$(echo "$(1)" | sed "s/-$(3)$$//")" $(1) ;\
}
endef
