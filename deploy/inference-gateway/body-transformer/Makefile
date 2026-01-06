# Body Transformer Makefile

IMAGE_NAME ?= body-transformer
IMAGE_TAG ?= latest
IMAGE ?= $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: all
all: build

.PHONY: build
build: ## Build the Go binary locally
	go build -o bin/body-transformer ./cmd

.PHONY: test
test: ## Run tests
	go test -v ./...

.PHONY: tidy
tidy: ## Run go mod tidy
	go mod tidy

.PHONY: image-build
image-build: ## Build Docker image
	docker build -t $(IMAGE) .

.PHONY: image-load
image-load: image-build ## Build and load into minikube
	minikube image load $(IMAGE)

.PHONY: deploy
deploy: ## Deploy to Kubernetes
	kubectl apply -f deploy/deployment.yaml -n my-model
	kubectl apply -f deploy/kgateway-config.yaml

.PHONY: undeploy
undeploy: ## Remove from Kubernetes
	kubectl delete -f deploy/kgateway-config.yaml --ignore-not-found
	kubectl delete -f deploy/deployment.yaml -n my-model --ignore-not-found

.PHONY: logs
logs: ## View logs
	kubectl logs -f -l app=body-transformer -n my-model

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

