# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

KUBECTL_DOC_REF ?= $(shell cat components/kubectl-doc/VERSION)
KUBECTL_DOC_REPO ?= https://github.com/sttts/kubectl-doc.git
KUBECTL_DOC_CACHE ?= ../.cache/kubectl-doc/$(KUBECTL_DOC_REF)

ifdef KUBECTL_DOC_CHECKOUT
KUBECTL_DOC_SOURCE := $(KUBECTL_DOC_CHECKOUT)
KUBECTL_DOC_SOURCE_TARGET :=
else
KUBECTL_DOC_SOURCE := $(KUBECTL_DOC_CACHE)
KUBECTL_DOC_SOURCE_TARGET := kubectl-doc-source
endif

.PHONY: gen
gen: gen-kubectl-doc-api-reference

.PHONY: gen-kubectl-doc-api-reference
gen-kubectl-doc-api-reference: $(KUBECTL_DOC_SOURCE_TARGET)
	KUBECTL_DOC_CHECKOUT="$(KUBECTL_DOC_SOURCE)" python3 kubectl_doc.py gen

.PHONY: kubectl-doc-source
kubectl-doc-source:
	python3 kubectl_doc.py checkout --repo "$(KUBECTL_DOC_REPO)" --checkout "$(KUBECTL_DOC_SOURCE)" --ref "$(KUBECTL_DOC_REF)"

.PHONY: check-generated
check-generated: gen
	cd .. && git diff --exit-code -- docs/kubernetes/api-reference fern/kubectl-doc-schemas fern/components/kubectl-doc/schemaSources.generated.ts
