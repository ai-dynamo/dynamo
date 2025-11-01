# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Makefile for Fault Injection Service

# Configuration
REGISTRY ?= localhost:5000
VERSION ?= latest
NAMESPACE ?= fault-injection-system

# Image names
API_IMAGE = $(REGISTRY)/fault-injection-api:$(VERSION)
GPU_IMAGE = $(REGISTRY)/gpu-fault-injector:$(VERSION)
NETWORK_IMAGE = $(REGISTRY)/network-fault-injector:$(VERSION)
MONITORING_IMAGE = $(REGISTRY)/monitoring-agent:$(VERSION)

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: build
build: build-api build-gpu build-network build-monitoring ## Build all Docker images

.PHONY: build-api
build-api: ## Build API service image
	docker build -t $(API_IMAGE) -f api-service/Dockerfile api-service/

.PHONY: build-gpu
build-gpu: ## Build GPU fault injector image
	docker build -t $(GPU_IMAGE) -f agents/gpu-fault-injector/Dockerfile agents/gpu-fault-injector/

.PHONY: build-network
build-network: ## Build network fault injector image
	docker build -t $(NETWORK_IMAGE) -f agents/network-fault-injector/Dockerfile agents/network-fault-injector/

.PHONY: build-monitoring
build-monitoring: ## Build monitoring agent image
	docker build -t $(MONITORING_IMAGE) -f agents/monitoring-agent/Dockerfile agents/monitoring-agent/

.PHONY: push
push: ## Push all images to registry
	docker push $(API_IMAGE)
	docker push $(GPU_IMAGE)
	docker push $(NETWORK_IMAGE)
	docker push $(MONITORING_IMAGE)

.PHONY: deploy
deploy: ## Deploy all components to Kubernetes
	kubectl apply -k deploy/

.PHONY: undeploy
undeploy: ## Remove all components from Kubernetes
	kubectl delete -k deploy/

.PHONY: logs-api
logs-api: ## View API service logs
	kubectl logs -n $(NAMESPACE) -l app=fault-injection-api -f

.PHONY: logs-gpu
logs-gpu: ## View GPU fault injector logs
	kubectl logs -n $(NAMESPACE) -l app=gpu-fault-injector -f

.PHONY: logs-network
logs-network: ## View network fault injector logs
	kubectl logs -n $(NAMESPACE) -l app=network-fault-injector -f

.PHONY: logs-monitoring
logs-monitoring: ## View monitoring agent logs
	kubectl logs -n $(NAMESPACE) -l app=monitoring-agent -f

.PHONY: port-forward
port-forward: ## Port-forward API service to localhost:8080
	kubectl port-forward -n $(NAMESPACE) svc/fault-injection-api 8080:8080

.PHONY: status
status: ## Check deployment status
	@echo "=== Namespace ==="
	kubectl get ns $(NAMESPACE)
	@echo "\n=== Pods ==="
	kubectl get pods -n $(NAMESPACE)
	@echo "\n=== Services ==="
	kubectl get svc -n $(NAMESPACE)
	@echo "\n=== DaemonSets ==="
	kubectl get ds -n $(NAMESPACE)

.PHONY: test
test: ## Run example tests (requires port-forwarding)
	cd examples && python -m pytest test_gpu_health_with_api.py -v -s

.PHONY: install-client
install-client: ## Install Python client library
	cd client && pip install -e .

.PHONY: clean
clean: ## Clean up local build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

.PHONY: all
all: build push deploy ## Build, push, and deploy all components

