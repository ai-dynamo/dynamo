# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Variables
DOCKER_SERVER ?= my-registry
IMAGE_TAG ?= latest
RUST_TARGET_DIR ?= target
PYTHON_VERSION ?= 3.12

# Docker image names
DYNAMO_BASE_IMAGE ?= $(DOCKER_SERVER)/dynamo-base:$(IMAGE_TAG)
DYNAMO_OPERATOR_IMAGE ?= $(DOCKER_SERVER)/dynamo-operator:$(IMAGE_TAG)

.PHONY: help
help: ## Display this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Development
.PHONY: clean
clean: ## Clean build artifacts
	cargo clean
	rm -rf dist/
	rm -rf lib/bindings/python/dist/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

.PHONY: install-deps
install-deps: ## Install development dependencies
	@echo "Installing Rust toolchain..."
	@if ! command -v rustup >/dev/null 2>&1; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		source $$HOME/.cargo/env; \
	fi
	@echo "Installing UV..."
	@if ! command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

##@ Build
.PHONY: build-rust
build-rust: ## Build Rust components
	@echo "Building Rust components..."
	cargo build --release --locked --features llamacpp,cuda

.PHONY: build-python-runtime
build-python-runtime: build-rust ## Build ai-dynamo-runtime Python wheel
	@echo "Building ai-dynamo-runtime wheel..."
	mkdir -p deploy/sdk/src/dynamo/sdk/cli/bin/
	rm -f deploy/sdk/src/dynamo/sdk/cli/bin/*
	ln -sf $(PWD)/$(RUST_TARGET_DIR)/release/dynamo-run deploy/sdk/src/dynamo/sdk/cli/bin/dynamo-run
	cd lib/bindings/python && uv build --wheel --out-dir ../../../dist --python $(PYTHON_VERSION)

.PHONY: build-python-main
build-python-main: build-python-runtime ## Build ai-dynamo Python wheel
	@echo "Building ai-dynamo wheel..."
	uv build --wheel --out-dir dist --python $(PYTHON_VERSION)

.PHONY: build-wheels
build-wheels: build-python-main ## Build all Python wheels
	@echo "All wheels built in dist/"
	@ls -la dist/

.PHONY: build-operator
build-operator: ## Build operator binary
	@echo "Building operator..."
	cd deploy/cloud/operator && CGO_ENABLED=0 go build -o bin/manager ./cmd/main.go

.PHONY: build
build: build-wheels build-operator ## Build all components

##@ Docker
.PHONY: docker-build
docker-build: ## Build main Docker image
	@echo "Building dynamo base Docker image..."
	docker build -t $(DYNAMO_BASE_IMAGE) -f Dockerfile .

.PHONY: docker-build-operator
docker-build-operator: ## Build operator Docker image
	@echo "Building operator Docker image..."
	docker build -t $(DYNAMO_OPERATOR_IMAGE) -f deploy/cloud/operator/Dockerfile deploy/cloud/operator/

.PHONY: docker-push
docker-push: ## Push main Docker image
	docker push $(DYNAMO_BASE_IMAGE)

.PHONY: docker-push-operator
docker-push-operator: ## Push operator Docker image
	docker push $(DYNAMO_OPERATOR_IMAGE)

.PHONY: docker-all
docker-all: docker-build docker-build-operator ## Build all Docker images

.PHONY: docker-push-all
docker-push-all: docker-push docker-push-operator ## Push all Docker images

##@ Testing
.PHONY: test-rust
test-rust: ## Run Rust tests
	cargo test

.PHONY: test-python
test-python: ## Run Python tests
	@echo "Running Python tests..."
	uv run pytest tests/

.PHONY: test-operator
test-operator: ## Run operator tests
	cd deploy/cloud/operator && make test

.PHONY: test
test: test-rust test-python test-operator ## Run all tests

##@ Linting
.PHONY: lint-rust
lint-rust: ## Lint Rust code
	cargo clippy -- -D warnings
	cargo fmt --check

.PHONY: lint-python
lint-python: ## Lint Python code
	uv run ruff check .
	uv run ruff format --check .

.PHONY: lint-operator
lint-operator: ## Lint operator Go code
	cd deploy/cloud/operator && make lint

.PHONY: lint
lint: lint-rust lint-python lint-operator ## Run all linting

##@ Formatting
.PHONY: format-rust
format-rust: ## Format Rust code
	cargo fmt

.PHONY: format-python
format-python: ## Format Python code
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: format
format: format-rust format-python ## Format all code

##@ CI/CD
.PHONY: ci-build
ci-build: clean lint test build docker-all ## Full CI build pipeline

.PHONY: ci-release
ci-release: ci-build docker-push-all ## Full release pipeline

##@ Development shortcuts
.PHONY: dev-setup
dev-setup: install-deps ## Set up development environment
	@echo "Creating Python virtual environment..."
	uv venv .venv --python $(PYTHON_VERSION)
	@echo "Installing Python dependencies..."
	uv pip install -r container/deps/requirements.txt
	@echo "Development environment ready!"
	@echo "Activate with: source .venv/bin/activate"

.PHONY: quick-build
quick-build: build-rust build-python-main ## Quick build for development

# Variables for override
EARTHLY_ARGS ?=

##@ Migration helpers (temporary)
.PHONY: earthly-test
earthly-test: ## Run tests using current Earthly (for comparison)
	@echo "WARNING: This is deprecated. Use 'make test' instead."
	earthly +all-test $(EARTHLY_ARGS)

.PHONY: earthly-docker
earthly-docker: ## Build Docker using current Earthly (for comparison)
	@echo "WARNING: This is deprecated. Use 'make docker-all' instead."
	earthly +all-docker $(EARTHLY_ARGS)
