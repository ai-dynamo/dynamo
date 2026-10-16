<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Kubernetes Helm Charts

The following Helm chart is available for the Dynamo Kubernetes Platform:

- [platform](./charts/platform/README.md) - This chart installs the complete Dynamo Kubernetes Platform, including the Dynamo Operator, NATS, etcd, Grove, and Kai Scheduler.

## CRD Management

The cluster-wide operator manages Custom Resource Definitions (CRDs) automatically. CRD manifests
are generated under [`deploy/operator/config/crd/bases`](../operator/config/crd/bases/) and bundled
in the operator image.

- **Initial installation and upgrades**: The operator Deployment's `crd-apply` init container applies
  CRDs from the operator image using server-side apply before the manager starts.
- **External management**: Set `dynamo-operator.upgradeCRD=false` when another process manages CRDs.
  With this setting, the chart installs no CRDs. Apply them separately before starting a cluster-wide
  operator. Namespace-restricted operators require this setting and use the CRDs managed by the
  cluster-wide operator.
