<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EPP configuration compatibility types

This package contains a Dynamo-owned adaptation of the upstream Gateway API
Inference Extension `EndpointPickerConfig` schema. Dynamo owns this copy so its
versioned API can preserve the legacy saturation-detector shape while emitting
the configuration expected by the Go EPP.

## Upstream provenance

The base schema was copied from the following upstream source:

- Repository: <https://github.com/kubernetes-sigs/gateway-api-inference-extension>
- Release branch: `release-1.5`
- Tag: `refs/tags/v1.5.0`
- Commit: `df9f8909fe06fe2dfd4a6fc0bd0c01a0acabf849`
- File: `apix/config/v1alpha1/endpointpickerconfig_types.go`

The deprecated `queueDepthThreshold`, `kvCacheUtilThreshold`, and
`metricsStalenessThreshold` fields were retained from:

- Release branch: `release-1.2`
- Tag: `refs/tags/v1.2.0`
- Commit: `52ed3abfd4aa2374cc0f4ab5abe94675eb35311c`
- File: `apix/config/v1alpha1/endpointpickerconfig_types.go`

## Dynamo modifications

Dynamo combines the GAIE v1.5 `pluginRef` field and the deprecated v1.2
threshold fields in one typed API structure. CRD validation makes the two forms
mutually exclusive. The operator preserves the submitted shape in the Dynamo
API and normalizes deprecated fields to a GAIE `utilization-detector` plugin
only when rendering the EPP ConfigMap.

The upstream file schema serializes an unset optional `dataLayer` as `null`.
The Dynamo type adds `omitempty` because Kubernetes rejects `null` for the
embedded CRD object field.

Unchanged nested configuration structures continue to use the types from the
GAIE v1.5 Go module. The adapted Go files retain the upstream Kubernetes
Authors copyright and Apache 2.0 license and identify NVIDIA modifications.
