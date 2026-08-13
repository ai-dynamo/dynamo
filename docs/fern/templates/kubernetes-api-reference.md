---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Kubernetes API References
subtitle: Generate the full CRD schema and maintain curated DGD, DGDR, and DCD field references.
---

Dynamo has two Kubernetes API reference surfaces:

- The [full Kubernetes API reference](../pages/reference/kubernetes-api/full-api-reference.mdx) is
  generated and exhaustive. It covers all API versions, custom resources, nested types, validation,
  and operator configuration.
- The curated [DGD](../pages/reference/kubernetes-api/dynamo-graph-deployment.mdx),
  [DGDR](../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx), and
  [DCD](../pages/reference/kubernetes-api/dynamo-component-deployment.mdx) pages are manually
  maintained user-facing views of the supported `nvidia.com/v1beta1` resources.

The curated pages use the full reference as a schema inventory, but they are **not mechanically
extracted from it**. The Kubernetes generator writes only `full-api-reference.mdx`. Update a curated
page manually when the resource it documents changes.

For the page skeleton, start from the
[Kubernetes API reference template](reference/kubernetes-api-reference.mdx).

## Source of Truth

Use this order when sources disagree:

1. Go API types, comments, and Kubebuilder validation markers under `deploy/operator/api/` define the
   CRD schema.
2. Controller and webhook code define runtime behavior that the schema cannot express, including
   reconciliation, immutability, precedence, and injected defaults.
3. The generated raw Markdown and full MDX reference reflect the Go schema.
4. The curated DGD, DGDR, and DCD pages explain the user-facing subset and link it to operational
   behavior.

Do not treat a curated page as the source for a field type, required marker, default, enum, minimum,
maximum, or pattern. Confirm those facts against the regenerated full reference or Go source.

## Generate the Full Reference

The generation pipeline has two explicit stages.

### Generate the raw CRD reference

Run:

```bash
make -C deploy/operator generate-api-docs
```

This target:

1. Runs `crd-ref-docs` against `deploy/operator/api/` using
   `deploy/operator/docs/crd-ref-docs-config.yaml`.
2. Concatenates `deploy/operator/docs/header.md`, the generated schema, and
   `deploy/operator/docs/footer.md`.
3. Writes the result to
   `docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md`.
4. Runs `deploy/operator/docs/fix-api-anchors.py` to disambiguate type anchors shared by multiple API
   versions and remove known unresolved type links.

The footer contains operator-injected defaults and implementation notes that do not come directly
from CRD schema types.

### Render the full Fern page

Run:

```bash
python3 docs/fern/scripts/gen_kubernetes_api.py
```

The renderer parses `api-reference-k8s.md` and writes
`docs/fern/pages/reference/kubernetes-api/full-api-reference.mdx`. It converts packages and types to
Fern cards, accordions, parameter fields, badges, and links while preserving validation details.

Do not edit either generated file by hand. Change the Go types, generator configuration, header,
footer, anchor fixer, parser, or renderer that owns the output.

## Update a Curated Resource Page

After regenerating the full reference:

1. Find the resource's `Spec`, `Status`, and referenced nested types in the full reference.
2. Update the curated page's field inventory, types, required markers, defaults, allowed values,
   validation, and deprecation state.
3. Add user-facing semantics from controller or webhook behavior when the schema description is not
   sufficient.
4. Keep shared fields in their owning page. For example, DGD links to the DCD shared component spec
   instead of reproducing every component field.
5. Add or update the minimal example and `kubectl` inspection commands when the resource workflow
   changes.
6. Check every internal type link, external Kubernetes type link, and heading anchor.

The curated page should be easier to navigate than the full reference, but it must not omit a
supported user-configurable field in its declared scope.

## Structure a Curated Resource Page

Use these sections when they apply:

1. Resource purpose and API version
2. Minimal YAML example
3. `## Spec Reference`
4. Resource-specific subsections for large domains such as Planner configuration
5. `## Status`
6. `## Additional Types`
7. Lifecycle, conditions, known issues, or compatibility notes when needed
8. `## Inspect a <Resource>`
9. `## Related Pages`

Do not expand standard `apiVersion`, `kind`, or `metadata` fields unless Dynamo adds a
resource-specific constraint or behavior.

## Indent Nested Fields

Use `<Indent>` to show object ownership, not merely because a field's type is complex.

The curated pages currently use at most **two nested `<Indent>` levels**:

| Presentation level | Meaning | Example |
|---|---|---|
| No `<Indent>` | Field at the current section root | `restart` under Spec or `status.rollingUpdate` under Status |
| One `<Indent>` | Direct child of the preceding object | `strategy` under `restart` |
| Two `<Indent>` levels | Grandchild of the section-root object | `type` under `restart.strategy` |

```mdx
<ParamField path="restart" type="Restart">
  Restart policy for the graph.
</ParamField>

<Indent>
  <ParamField path="strategy" type="RestartStrategy">
    How components are restarted.
  </ParamField>

  <Indent>
    <ParamField path="type" type="RestartStrategyType" default="Sequential">
      Restart execution mode.
    </ParamField>
  </Indent>
</Indent>
```

Follow these path conventions:

- Under `## Spec Reference`, omit the redundant `spec.` prefix from root fields.
- Under `## Status`, include `status.` on root fields, such as `status.conditions`.
- Inside `<Indent>`, use the local field name (`strategy`, then `type`) rather than repeating the
  dotted path.
- Under `## Additional Types`, use local field names because the type heading establishes the
  parent.

Do not add a third `<Indent>` level. Instead, move the deeper shape to `## Additional Types` or link
to the page that owns it. This is a presentation limit, not a Kubernetes schema limit.

Break a type out under `## Additional Types` when it is reused, appears as a list or map value, has
many fields, or would push the main Spec or Status tree beyond two indentation levels. Do not indent
array elements solely because the parent type is a list.

## Link Internal and External Types

The `type` property on `<ParamField>` is plain text and cannot contain a Markdown link. Keep the
qualified type in the property, then add the link in the field body.

### Internal Dynamo types

Link to the local type under `## Additional Types` or to the sibling page that owns the type:

```mdx
<ParamField path="status.components" type="map[string]ComponentReplicaStatus">
  Per-component replica status.

  See [ComponentReplicaStatus](#componentreplicastatus).
</ParamField>
```

Do not repeat a shared type only to avoid a cross-page link.

### External Kubernetes types

Do not expand upstream Kubernetes types under `## Additional Types`. Link to the canonical Go package
documentation immediately after the field description:

```mdx
<ParamField path="startTime" type="metav1.Time">
  When the rolling update began.

  <a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Time" target="_blank">metav1.Time</a>
</ParamField>
```

Preserve the qualified type used in the curated page. For arrays and maps, link the element or value
type once.

| Type prefix or example | Canonical package documentation |
|---|---|
| `metav1.Time`, `metav1.Condition` | `https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1` |
| `core/v1.EnvVar`, `core/v1.PodTemplateSpec` | `https://pkg.go.dev/k8s.io/api/core/v1` |
| `batch/v1.JobSpec` | `https://pkg.go.dev/k8s.io/api/batch/v1` |
| `runtime.RawExtension` | `https://pkg.go.dev/k8s.io/apimachinery/pkg/runtime` |
| `resource.Quantity` | `https://pkg.go.dev/k8s.io/apimachinery/pkg/api/resource` |

Use the symbol anchor at the end of the URL, such as `#Time`, `#Condition`, or `#PodTemplateSpec`.
The generated full reference may use Kubernetes' generated API documentation links instead; keep
those generator-owned links intact.

## Record Field Metadata

Map schema facts to Fern components consistently:

- Required field: `required={true}`
- Default: `default="value"`
- Deprecated field: `deprecated={true}` plus a replacement or migration statement
- Enum: list every allowed value, preferably with the established Badge treatment
- Validation: state minimum, maximum, pattern, list bounds, mutual exclusion, or immutability in
  prose
- Runtime precedence or lifecycle behavior: verify against controller or webhook code, not only the
  CRD schema

Keep descriptions user-facing. Do not paste raw Go comments when they are repetitive, implementation
focused, or missing important operational behavior.

## Validate the References

Run the generation and focused tests:

```bash
make -C deploy/operator generate-api-docs
python3 docs/fern/scripts/gen_kubernetes_api.py
python3 -m pytest -q \
  docs/fern/scripts/tests/test_gen_kubernetes_api.py \
  docs/fern/scripts/tests/test_api_reference_regressions.py
```

Then run the normal documentation checks:

```bash
cd docs/fern
fern check
fern docs broken-links
```

Review the diff for both generated files and every affected curated page. Confirm that no generated
page was hand-edited and that the curated reference still matches the current `v1beta1` surface.
