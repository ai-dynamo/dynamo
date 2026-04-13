# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

IMG ?= nvcr.io/nvidian/dynamo-dev/gpu-memory-service:latest
CONTAINER_TOOL ?= docker
RUNTIME_IMAGE_PLATFORM ?= linux/amd64
DOCKER_BUILD_ARGS ?=

REPO_ROOT := $(abspath $(CURDIR)/../..)
DOCKERFILE := $(CURDIR)/Dockerfile

.PHONY: docker-build
docker-build:
	$(CONTAINER_TOOL) build --platform $(RUNTIME_IMAGE_PLATFORM) $(DOCKER_BUILD_ARGS) -f $(DOCKERFILE) -t $(IMG) $(REPO_ROOT)

.PHONY: docker-push
docker-push:
	$(CONTAINER_TOOL) push $(IMG)
