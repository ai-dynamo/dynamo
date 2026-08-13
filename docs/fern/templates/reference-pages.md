---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Reference Pages
subtitle: Document complete configuration surfaces and exact technical contracts.
---

Reference pages provide structured lookup material. Readers should be able to find an exact field,
flag, API, environment variable, default, or contract without following a workflow.

For help choosing another content type, see the
[Documentation Content Guidelines](docs-guidelines.md).

## Cover the Complete Surface

Define a clear scope, then cover the full supported interface within it. Include the details that
control how readers use the interface correctly:

- Types and defaults
- Allowed values and validation rules
- Required and optional fields
- Precedence and interactions
- Version, compatibility, or stability constraints
- Errors and observable behavior when relevant

Reference coverage should resemble the completeness of a command's `--help` output or a system
manual page, while remaining organized for the interface being documented.

## Organize for Lookup

Group references by interface or category, such as Python, Rust, Kubernetes, components, or
backends. Use consistent entry structure and terminology across related pages.

Use Fern parameter components such as `<ParamField>` for fields, arguments, and environment
variables when appropriate. Keep examples short and illustrative, and link to a
[user-facing guide](user-facing-guides.md) for end-to-end usage.

## Start from the API Template

Use the [code API reference template](reference/api-reference.mdx) for a manually authored
module, package, crate, class, function group, or protocol surface. Its reusable unit is a public
symbol or operation: signature, parameters, return value, errors, example, and source link. Repeat
the symbol block when one page documents a related API group, and remove class-only sections for
functions or other smaller surfaces.

Use the [Kubernetes API reference template](reference/kubernetes-api-reference.mdx) for a curated
custom-resource page. Kubernetes resources need a separate structure for desired state under Spec,
controller-observed state under Status, reusable nested types, and `kubectl` inspection. Follow
[Writing Kubernetes API References](kubernetes-api-reference.md) for the generation pipeline,
indentation limit, type-link rules, and curated-page update process.

Do not force every API shape into either template:

- HTTP APIs backed by an API definition should use Fern's `EndpointSchemaSnippet`,
  `EndpointRequestSnippet`, and `EndpointResponseSnippet` components.
- Large generated language APIs should encode the same information in their renderer rather than
  copying a manual page template.
- A page with a generated-file marker must be changed through its source or generator.

## Prefer Generated References

Generate reference content when the source contract can produce it reliably. Do not hand-edit pages
that carry a generated-file marker. Update the source and rerun the corresponding generator instead.

The Python and Rust API pages and the full Kubernetes API reference are generated. Their workflows
are documented in [`docs/fern/AGENTS.md`](../AGENTS.md).
