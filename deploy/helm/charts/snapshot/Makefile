# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

HELM_UNITTEST_VERSION ?= 0.7.2

.PHONY: helm-unittest-install
helm-unittest-install:
	@helm plugin list | grep -q unittest || \
		helm plugin install https://github.com/helm-unittest/helm-unittest --version $(HELM_UNITTEST_VERSION) --verify=false

.PHONY: lint
lint:
	helm lint .

.PHONY: test
test: helm-unittest-install
	helm unittest .
