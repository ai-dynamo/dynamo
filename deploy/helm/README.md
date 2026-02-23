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

CRDs are bundled in the operator subchart's `crds/` directory and managed automatically:

- **Initial install**: Helm natively installs CRDs from the `crds/` directory during `helm install`.
- **Upgrades**: A `pre-upgrade` hook Job applies CRDs using server-side apply from the operator image. This is necessary because Helm does not update CRDs from the `crds/` directory on `helm upgrade`. This can be disabled by setting `upgradeCRD: false`.