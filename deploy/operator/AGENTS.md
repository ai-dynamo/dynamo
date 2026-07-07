<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- RBAC changes for operator code must update the `+kubebuilder:rbac` markers.
- Run `make manifests` to regenerate both `config/rbac/role.yaml` and the
  platform chart's `../helm/charts/platform/components/operator/files/manager-rules.yaml`.
- Keep chart-only grants in the manual section of the platform chart's
  `../helm/charts/platform/components/operator/templates/manager-rbac.yaml`.
