<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- RBAC changes for operator code must update the `+kubebuilder:rbac` markers.
- Run `make manifests` to regenerate both `config/rbac/role.yaml` and the
  platform chart's `../helm/charts/platform/components/operator/files/manager-role.yaml`.
- Keep chart-only grants in the manual section of the platform chart's
  `../helm/charts/platform/components/operator/templates/manager-rbac.yaml`.

## Go Test Style

- Use `t.Log` to tell the test's story, with one heading before each block that
  implements a test step. Preserve test paragraph comments by converting them
  to `t.Log` headings when moving or refactoring the code.
- Keep test fixtures in local variables or constants and owned by one test.
- Avoid hiding bespoke test logic in closures. Reserve closures for standard
  helpers such as `Eventually`.
- Prefer a shared construction DSL over shared fixture objects when setup is
  complex or repetitive; the DSL should describe the object at a high level.
- Prefer table tests over one-off tests and do not duplicate behavior already
  covered by a table.
