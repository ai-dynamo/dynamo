<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recipe Contribution Guide

Use `nvidia.com/v1beta1` for new recipes. `nvidia.com/v1alpha1` is deprecated; use alpha only when maintaining an existing alpha recipe. Kustomize patches must target the base manifest's API and cannot convert between the alpha and beta shapes.

Review rejects a new or refreshed base recipe whose standard role names deviate from the canon. Aggregate bases use `Frontend` and `Worker`. Disaggregated bases use `Frontend`, `PrefillWorker`, and `DecodeWorker`. Beta bases keep that literal order and place optional components afterward. Review the complete [template field contract](templates/README.md#field-contract) before contributing.

Cluster owners can copy and fill the [beta cluster Kustomization starter](templates/kustomize/README.md) once per cluster. Filled cluster values remain outside the portable recipe contribution.

## Choose a Contribution Path

| Goal | Workflow |
| --- | --- |
| Author a new portable recipe | Start from the closest [representative example](templates/README.md#choose-a-template), follow the [field contract](templates/README.md#field-contract), and submit a standalone `deploy.yaml`. |
| Adapt portable beta recipes to one cluster | Copy and privately fill the [cluster Kustomization starter](templates/kustomize/README.md); do not commit the filled copy upstream. |
| Publish multiple maintained public variants of one recipe | Use the [Kustomize matrix workflow](#publish-public-kustomize-variants). A matrix may use a template-derived manifest as `kustomize/base/deploy.yaml`. |

Adapting a recipe for one target cluster does not by itself require a public matrix. Keep site-specific configuration in the cluster-owned Kustomization.

## Author a Portable Recipe

### Responsibilities

| Owner | Responsibility |
| --- | --- |
| Cluster owner | Copies and maintains the [cluster Kustomization starter](templates/kustomize/README.md) for each cluster. Owns namespace selection, scheduling, provisioning and site-specific identities for PVCs, optional cluster-wide startup-probe policy, registry credentials, provider networking, and physical ComputeDomain or Dynamic Resource Allocation (DRA) realization. Keeps filled site values outside the recipe contribution. |
| Recipe developer | Selects and adapts a representative example. Owns portable serving intent, iterates through the cluster owner's Kustomization, and submits the portable base. |

The recipe developer obtains the applicable Kustomization from the cluster owner and uses it to qualify each portable base. Keep the filled cluster configuration outside the recipe contribution.

### Select and Copy an Example

Follow the catalog's [template selection guidance](templates/README.md#choose-a-template), then copy the selected example to the new recipe directory as `deploy.yaml`. This filename is the repository convention for a standalone recipe.

Use the following standard structure:

```text
<model-name>/
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
├── <framework>/
│   └── <deployment-mode>/
│       ├── deploy.yaml
│       └── perf.yaml (optional)
└── README.md (optional)
```

The portable base belongs at:

```text
recipes/<model>/<framework>/<mode>/deploy.yaml
```

### Adapt the Base

Apply the catalog's [field contract](templates/README.md#field-contract):

- Preserve exact component names, beta ordering, coordinated reference bundles, and patch-hook positions. For components selected by `cache-binding`, keep the `shared-model-cache` volume mount and volume first in their respective lists.
- Edit the model, framework configuration, image, parallelism, GPU shape, memory, transfer configuration, and engine settings as one coherent runtime bundle.
- Retain the canonical `shared-model-cache` bundle on backend workers when applicable, including the `/shared-model-cache` container path, but leave its cluster-specific physical identity and provisioning out of the base. Beta Frontends retrieve model metadata from workers and do not mount the cache.
- Omit credential Secret references. Beta recipes run offline against the pre-populated cache and set `HF_HUB_OFFLINE: "1"` and `TRANSFORMERS_OFFLINE: "1"` on every component.
- Use `imagePullPolicy: IfNotPresent` on every container. Runtime containers use `command: [python3]` and token-list arguments beginning with `-m` and `dynamo.<module>`. Use Kubernetes `$(VAR)` argument substitution without a shell prelude.
- Keep the standard beta backend-worker security context: UID and GID `0`, with `IPC_LOCK`, `SYS_PTRACE`, and `SYS_RESOURCE` capabilities. Frontend containers omit this security context.
- Rely on the operator's complete probes by default. A recipe may provide a complete probe only when it deliberately tightens an operator budget, documents the default it replaces, and restates every field it needs; a cluster-wide worker startup adjustment instead uses the optional `probes` Component.
- Leave cluster-supplied scheduling, registry credentials, provider networking, host bindings, namespace, and physical DRA settings out of the base.

Scalar tuning, long-context profiles, router behavior, and ordinary multi-node workers usually belong in the copied recipe instead of a new catalog example. See [Adapt Without Adding a Template](templates/README.md#adapt-without-adding-a-template).

### Iterate Through the Cluster Kustomization

Reference the portable `deploy.yaml` from the cluster-owned Kustomization. The starter pins standalone Kustomize v5.8.1 and documents its validation command and private fill-in contract. Use the following feedback loop:

1. Render the composition and inspect the resulting manifest.
2. Run a server-side dry run against a cluster with the required CRDs and policies.
3. Apply the composition.
4. Observe admission, scheduling, readiness, responses, and relevant performance behavior.
5. Update the portable base and repeat until it is ready for review.

For example, after filling the starter and setting `NAMESPACE`, run from the
Dynamo repository root:

```bash
python3 scripts/validate-recipe-kustomization.py \
  <portable-deploy.yaml> \
  <cluster-kustomization>/kustomization.yaml \
  --kustomize-bin "$(command -v kustomize)"
kustomize build --load-restrictor LoadRestrictionsNone \
  <cluster-kustomization> | \
  kubectl apply --dry-run=server -f - -n "$NAMESPACE"
kustomize build --load-restrictor LoadRestrictionsNone \
  <cluster-kustomization> | \
  kubectl apply -f - -n "$NAMESPACE"
```

Using the standalone renderer ensures the applied manifest has the same Kustomize version used by validation. The load-restrictor option supports an explicitly referenced portable base outside the copied scaffold; review every such path. Use `kubectl apply -k` only when every reference is within kubectl's permitted load roots and `kubectl version --client --output=yaml` reports an embedded Kustomize v5.8.1. Rendering and admission checks do not replace runtime qualification on the intended hardware and network.

### Ship the Portable Base

Submit the portable base as `recipes/<model>/<framework>/<mode>/deploy.yaml`. Do not submit a filled cluster Kustomization or output rendered from site-specific values. Site-neutral shared Components are repository source and may be contributed when they are reusable.

Public matrix variants are the documented exception: commit their generated overlay `kustomization.yaml` files and `deploy-<name>.yaml` manifests as described below.

## Publish Public Kustomize Variants

Use this workflow when repository users need multiple maintained public provider
or network variants of one deployment shape. Every selected Component must target
the base manifest's API and field paths. The existing public provider matrix
examples use the legacy alpha shape: their static and template-generated
Components target `spec.services` and cannot convert or patch a beta
`spec.components` base. The cluster-owned [beta cluster
starter](templates/kustomize/README.md) instead uses guarded JSON 6902 patches
against canonical beta component positions and does not include the central
OpenAPI Component.

Recipe-local bases, Components, and generated public overlays live under
`<deployment>/kustomize/`. Shared Components reusable by multiple recipes live
under `recipes/kustomize/components/`. Contributor-only Jinja template sources
for this matrix workflow live under `recipes/kustomize/templates/`; they are
separate from the portable examples and beta cluster starter under
`recipes/templates/`. Run the commands in this guide from the repository root.
Keep the checked-in manifests directly applicable and easy to review:

```text
<deployment>/
├── .kustomize-matrix.yaml
├── deploy-generic.yaml
├── deploy-aws-p5.48xlarge.yaml
├── deploy-gcp-roce.yaml
├── perf.yaml
└── kustomize/
    ├── base/
    │   ├── deploy.yaml
    │   └── kustomization.yaml
    ├── components/ (optional)
    │   └── <recipe-specific-building-block>/
    └── overlays/
        ├── generic/
        │   └── kustomization.yaml
        ├── aws-p5.48xlarge/
        │   └── kustomization.yaml
        ├── gcp-roce/
        │   └── kustomization.yaml
```

Kustomize is both the authoring model and the documentation of a variant: the base and Components explain individual settings, while each checked-in public overlay documents the selected composition. The rendered `deploy-<name>.yaml` is the exact, fully materialized result.

### Use a Variant

Recipe users may apply a checked-in rendered manifest directly:

```bash
kubectl apply -f <deployment>/deploy-<name>.yaml -n ${NAMESPACE}
```

They may instead inspect or apply the checked-in public Kustomization, which documents the base and selected Components:

```bash
kubectl apply -k <deployment>/kustomize/overlays/<name> -n ${NAMESPACE}
```

Users can also create an uncommitted `kustomization.yaml` in the repository checkout and apply it with `kubectl apply -k`. For an ad hoc composition without creating a directory, `compose` creates a temporary Kustomization and writes the real Kustomize output to stdout. Its target comes first, followed by Components and then Kustomize build options:

```bash
scripts/kustomize-matrix.py compose \
  <target-kustomization> \
  <component-path>... \
  | kubectl apply -f - -n ${NAMESPACE}
```

None of these user workflows requires `unfold` or `render`.

### Contribute a Variant

For a matrix-backed recipe, the source of truth is
`.kustomize-matrix.yaml`, the recipe-local `kustomize/base/`, optional
`kustomize/components/`, template sources, and any referenced Components under
`recipes/kustomize/components/`. The generated files are public overlay
`kustomization.yaml` files, their generated `components/` template Components,
`deploy-<name>.yaml` manifests, and the central
`recipes/kustomize/components/dynamo-openapi/dynamo-openapi.json` schema. Commit
the generated files for users to inspect and apply, but do not edit them by hand.

The render convention is:

- `kustomize/base/` is shared input and is not rendered directly.
- `kustomize/overlays/<name>/` renders to `deploy-<name>.yaml`.
- `kustomize/overlays/generic/` renders to `deploy-generic.yaml`. Use it when a
  generic deployable variant exists.
- `kustomize/components/` is for recipe-specific Kustomize building blocks and is
  not rendered. Shared building blocks live under
  `recipes/kustomize/components/` and are also not rendered directly.
- `recipes/kustomize/templates/` holds contributor-only Jinja template sources.
  `unfold` renders each selected source into an ordinary Component under the
  public overlay's `components/` directory at the selected path. Users never
  need Jinja to inspect or apply that overlay.
- Legacy strategic-merge bases that patch Dynamo CRDs include the central
  `recipes/kustomize/components/dynamo-openapi/` Component. Its generated
  schema is derived from every operator CRD and lets strategic merge patches
  merge CRD map lists such as `env` by name. Guarded JSON 6902 Components,
  including the beta cluster starter, do not include it.
- The central `recipes/kustomize/components/disagg-workers/` Components apply
  to bases containing one DGD with backend-neutral `PrefillWorker` and
  `DecodeWorker` service keys.

Within the legacy alpha matrix path, prefer resource-shaped Kustomize merge patches where possible. For other Custom Resource Definition (CRD) list fields, include the complete intended list in the merge patch unless the schema supplies an OpenAPI merge key. Use guarded JSON 6902 for the beta cluster starter and beta Components whose correctness depends on canonical list positions and fail-loud preconditions.

Edit the Kustomize source, not the generated manifests. A recipe matrix is an
explicit `.kustomize-matrix.yaml` beside the recipe. It names the Kustomize
`source`, a `nameTemplate`, and matrix dimensions. Every dimension value has a
human-readable `name`, plus optional Kustomize `components`, `templates`, and
template `values`; output names interpolate only the value names, never their
paths:

```yaml
source: kustomize/base
nameTemplate: "${variant}"
matrix:
  variant:
    - name: aws-p5.48xlarge
      templates:
        - source: ../../../kustomize/templates/aws-efa/p5.48xlarge
          path: components/efa
      values:
        # Variant values override defaults from a selected template's values.yaml.
        EFAS_PER_GPU: 4
```

A template selection names a source directory relative to the matrix and an
output `path` relative to the generated overlay. The output path must be under
`components/`; `path: components/efa` produces a normal local Component at
`kustomize/overlays/<name>/components/efa/`. Template output paths selected by
one variant must be unique and must not contain one another. The selected
template directory or its parent must provide `kustomization.yaml` or
`kustomization.yaml.j2` and may provide a plain `values.yaml` mapping. Whether
plain or rendered from Jinja, the selected Kustomization must produce one
v1alpha1 Kustomize `Component`. A Jinja source receives `values` and an indexed
`base` rendered from the matrix `source`. `base` is indexed by lower-case Kind
and `metadata.name`, for example
`base.configmap[values.PREFILL_CONFIG]`. When exactly one resource of a Kind is
expected, use `base.dynamographdeployment | only`; this fails clearly if the
source changes to contain zero or multiple such resources. Templates may use
embedded patches or ordinary local Kustomize source paths.

#### Inheriting Template Files

Put common YAML files and defaults directly in the parent of concrete template
directories when several variants share the same Kustomize structure:

```text
aws-efa/
├── kustomization.yaml.j2
├── resources.yaml.j2
├── values.yaml
├── p5.48xlarge/
│   ├── kustomization.yaml.j2
│   └── values.yaml
└── p6-b200.48xlarge/
    └── values.yaml
```

The matrix selects one concrete directory:

```yaml
templates:
  - source: ../../../kustomize/templates/aws-efa/p5.48xlarge
    path: components/efa
```

`unfold` reads only direct `*.yaml` and `*.yaml.j2` files from the selected
directory's parent and then from the selected directory. It never scans either
directory recursively. Files are matched by their output name after removing a
final `.j2`; for example, a selected `resources.yaml` replaces an inherited
`resources.yaml.j2`, and a selected `resources.yaml.j2` likewise replaces an
inherited `resources.yaml`. It replaces complete files and does not merge YAML
content.
Values use shallow, top-level replacement in this order:

```text
parent values.yaml -> selected values.yaml -> matrix values
```

The effective files are rendered once into the selected output path. A
values-only selected directory inherits the parent's files unchanged. A
specialized directory can replace `kustomization.yaml.j2` or any other inherited
YAML file by providing the same output name. Subdirectories and non-YAML files
are never materialized. A template can reference shared Components or
resources; `unfold` rebases those external paths for the generated location.
Jinja rendering uses strict, immutable sandboxed values: undefined names and
attempts to mutate data are errors. Rendering a matrix that selects templates
requires `jinja2==3.1.6`, which is installed in the development and test
dependency sets.

Regenerate derived artifacts in order: `unfold` writes every checked-in public
overlay `kustomization.yaml` file and selected generated Component for the
matrix; `render` invokes Kustomize and writes every rendered
`deploy-<name>.yaml` manifest and the central CRD schema:

```bash
scripts/kustomize-matrix.py unfold <matrix.yaml>
scripts/kustomize-matrix.py render <matrix.yaml>
```

To inspect only one concrete public overlay without regenerating the matrix, run:

```bash
kustomize build <deployment>/kustomize/overlays/<name>
```

For dependent Components, use flat, explicit names such as `aws-efa` and
`vllm-disagg/aws-efa`. An instance-type template can include a generic fabric
Component and patch the hardware-specific resource count derived from the base
deployment.

`render` runs `kustomize build` and falls back to `kubectl kustomize` when `kustomize` is not on `PATH`. Kustomize drops comments while rendering Kubernetes objects, so the renderer re-inserts non-SPDX comments from the source YAML before matching rendered fields. It does not copy comments inside literal block scalars because those already render in place. It also refreshes the central OpenAPI schema from the operator CRDs.

`scripts/kustomize-matrix.py check` validates all generated overlays, manifests, and the schema; the Recipe Check CI job runs the same command. It also reports artifacts left by a moved matrix. Normal generation leaves those artifacts in place; after reviewing them, clean them explicitly:

```bash
scripts/kustomize-matrix.py unfold --clean <matrix.yaml>
scripts/kustomize-matrix.py render --clean <matrix.yaml>
```

### Validate a Matrix

Run the repository-wide freshness check after changing a matrix, base, Component, generated variant, or operator CRD:

```bash
python3 scripts/kustomize-matrix.py check
```

This check verifies matrix expansion and generated-artifact freshness. It does not qualify admission, readiness, responses, or performance.

## Validate Before Review

Before submitting a recipe contribution, confirm that:

- a new recipe uses `nvidia.com/v1beta1`;
- standard roles use the canonical names and beta order;
- the portable base follows the [field contract](templates/README.md#field-contract) and omits cluster-owned fields;
- the portable base omits credential Secret references and relies on operator probes unless a complete, documented per-recipe override is intentional;
- beta components retain the offline settings, backend workers retain the standard security context, and every container uses the catalog's exec form and image-pull policy;
- probe ownership belongs to either the base or the optional `probes` Component, never both;
- the target-cluster composition was rendered, checked with a server-side dry run, and exercised on its intended cluster;
- matrix-backed changes pass `python3 scripts/kustomize-matrix.py check`; and
- generated matrix files were regenerated, reviewed, and not hand-edited.

Static rendering, schema, and matrix checks do not prove runtime correctness or performance. Include the relevant target-cluster qualification when requesting review.
