<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Platform Chart

## Upgrade Preflight Checks

Follow [DEP #17395](https://github.com/ai-dynamo/dynamo/issues/17395) when adding or changing an
operator upgrade preflight check.

- Keep shared Job execution settings under `dynamo-operator.upgradePreflight.job` and
  check-specific policy under `dynamo-operator.upgradePreflight.checks.<checkName>`.
- Give every check one stable identifier. Represent it as lower camel case in values, kebab case in
  Kubernetes names and paths, and snake case in Python symbols. For example, `upgradePath`,
  `upgrade-path`, and `upgrade_path` identify the same check.
- Name each hook Job with the `check-<kebab-case-id>` suffix so its values key, implementation, test,
  and live Kubernetes resource are traceable to one another.
- Put each check's Helm template in `templates/upgrade-preflight/<check-id>.yaml`, executable code in
  `files/upgrade-preflight/checks/<check-id>/`, and tests in `tests/upgrade-preflight/` with the same
  identifier. Keep reusable upgrade runtime code in `files/upgrade-preflight/common/` and generic
  preflight code shared with non-upgrade checks in `files/preflight/`.
- Render common ServiceAccount, Job, Pod security, image, scheduling, deadline, cleanup, and log
  policy through the shared upgrade-preflight Helm helper. Do not copy the full Job scaffold into
  each check template.
- Give every check its own Job and least-privilege RBAC. Do not widen a narrow check's permissions
  or combine unrelated checks into a universal script or Job.
- Mandatory compatibility checks must not have persistent disable switches. Put any exceptional
  bypass below the affected check, scope it narrowly, and ensure it cannot bypass sibling checks.
- Update `values.yaml`, `README.md.gotmpl`, the generated `README.md`, Helm rendering tests, and
  Python unit tests together when the values or behavior of a check changes.

