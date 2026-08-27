<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recipe Contribution Guide

Use `nvidia.com/v1beta1` for new recipes. `nvidia.com/v1alpha1` is deprecated; use alpha only when maintaining an existing alpha recipe. Kustomize patches must target the base manifest's API and cannot convert between the alpha and beta shapes.

Review rejects a new or refreshed base recipe whose standard role names deviate from the canon. Aggregate bases use `Frontend` and `Worker`. Disaggregated bases use `Frontend`, `PrefillWorker`, and `DecodeWorker`. Beta bases keep that literal order and place optional components afterward. Review the complete [template field contract](templates/README.md#field-contract) before contributing.

## Choose a Contribution Path

| Goal | Workflow |
| --- | --- |
| Author a new portable recipe | Start from the closest [representative example](templates/README.md#choose-a-template), follow the [field contract](templates/README.md#field-contract), and submit a standalone `deploy.yaml`. |
| Publish multiple maintained public variants of one recipe | Use the [Kustomize matrix workflow](#publish-public-kustomize-variants). A matrix may use a template-derived manifest as `kustomize/base/deploy.yaml`. |

Adapting a recipe for one target cluster does not by itself require a public matrix. Keep site-specific configuration in the cluster-owned Kustomization.

## Author a Portable Recipe

### Responsibilities

| Owner | Responsibility |
| --- | --- |
| Cluster owner | Creates and maintains one reusable cluster Kustomization for each cluster. Owns namespace selection, scheduling, provisioning and site-specific identities for PVCs and application Secrets, registry credentials, provider networking, and physical ComputeDomain or Dynamic Resource Allocation (DRA) realization. Keeps filled site values outside the recipe contribution. |
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

- Preserve exact component names, beta ordering, coordinated reference bundles, and patch-hook positions.
- Edit the model, framework configuration, image, parallelism, GPU shape, memory, transfer configuration, and engine settings as one coherent runtime bundle.
- Retain canonical defaults such as `model-cache` and `hf-token-secret` when applicable, but leave their cluster-specific physical identities and provisioning out of the base.
- Leave cluster-supplied scheduling, registry credentials, provider networking, host bindings, namespace, and physical DRA settings out of the base.

Scalar tuning, long-context profiles, router behavior, and ordinary multi-node workers usually belong in the copied recipe instead of a new catalog example. See [Adapt Without Adding a Template](templates/README.md#adapt-without-adding-a-template).

### Iterate Through the Cluster Kustomization

Reference the portable `deploy.yaml` from the cluster-owned Kustomization. Use the following feedback loop:

1. Render the composition and inspect the resulting manifest.
2. Run a server-side dry run against a cluster with the required CRDs and policies.
3. Apply the composition.
4. Observe admission, scheduling, readiness, responses, and relevant performance behavior.
5. Update the portable base and repeat until it is ready for review.

For example:

```bash
kubectl kustomize <cluster-kustomization>
kubectl apply --dry-run=server -k <cluster-kustomization> -n <namespace>
kubectl apply -k <cluster-kustomization> -n <namespace>
```

Rendering and admission checks do not replace runtime qualification on the intended hardware and network.

### Ship the Portable Base

Submit the portable base as `recipes/<model>/<framework>/<mode>/deploy.yaml`. Do not submit a filled cluster Kustomization or output rendered from site-specific values. Site-neutral shared Components are repository source and may be contributed when they are reusable.

Public matrix variants are the documented exception: commit their generated overlay `kustomization.yaml` files and `deploy-<name>.yaml` manifests as described below.

## Publish Public Kustomize Variants

Use this workflow when repository users need multiple maintained public provider or network variants of one deployment shape. Every selected Component must target the base manifest's API and field paths. The current shared provider Components target alpha `spec.services`; they cannot convert or patch a beta `spec.components` base without beta-targeted alternatives.

Recipe-local bases, Components, and generated public overlays live under `<deployment>/kustomize/`. Shared Components reusable by multiple recipes live under `recipes/kustomize/components/`. Run the commands in this guide from the repository root. Keep the checked-in manifests directly applicable and easy to review:

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

For a matrix-backed recipe, the source of truth is `.kustomize-matrix.yaml`, the recipe-local `kustomize/base/`, optional `kustomize/components/`, and any referenced Components under `recipes/kustomize/components/`. The generated files are public overlay `kustomization.yaml` files, `deploy-<name>.yaml` manifests, and the central `recipes/kustomize/components/dynamo-openapi/dynamo-openapi.json` schema. Commit the generated files for users to inspect and apply, but do not edit them by hand.

The render convention is:

- `kustomize/base/` is shared input and is not rendered directly.
- `kustomize/overlays/<name>/` renders to `deploy-<name>.yaml`.
- `kustomize/overlays/generic/` renders to `deploy-generic.yaml`. Use it when a generic deployable variant exists.
- `kustomize/components/` is for recipe-specific Kustomize building blocks and is not rendered. Shared building blocks live under `recipes/kustomize/components/` and are also not rendered directly.
- Bases that patch Dynamo CRDs include the central `recipes/kustomize/components/dynamo-openapi/` Component. Its generated schema is derived from every operator CRD and lets strategic merge patches merge CRD map lists such as `env` by name.
- The central `recipes/kustomize/components/disagg-workers/` Components apply to bases containing one DGD with backend-neutral `PrefillWorker` and `DecodeWorker` service keys.

Prefer resource-shaped Kustomize merge patches over JSON patches where possible. For other Custom Resource Definition (CRD) list fields, include the complete intended list in the merge patch unless the schema supplies an OpenAPI merge key.

Edit the Kustomize source, not the generated manifests. A recipe matrix is an explicit `.kustomize-matrix.yaml` beside the recipe. It names the Kustomize `source`, a `nameTemplate`, and matrix dimensions. Every dimension value has a human-readable `name` and a list of Kustomize `components`; output names interpolate only the value names, never their paths:

```yaml
source: kustomize/base
nameTemplate: "${variant}"
matrix:
  variant:
    - name: aws-p5.48xlarge
      components:
        - ../../../kustomize/components/aws-efa-p16d16
```

Regenerate derived artifacts in order: `unfold` writes every checked-in public overlay `kustomization.yaml` file for the matrix; `render` invokes Kustomize and writes every rendered `deploy-<name>.yaml` manifest for the matrix and the central CRD schema:

```bash
scripts/kustomize-matrix.py unfold <matrix.yaml>
scripts/kustomize-matrix.py render <matrix.yaml>
```

To inspect only one concrete public overlay without regenerating the matrix, run:

```bash
kustomize build <deployment>/kustomize/overlays/<name>
```

For dependent Components, use flat, explicit names such as `aws-efa` and `aws-efa-p8d16`. A leaf Component may include its predecessor, while the matrix selects only the leaf.

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
- the target-cluster composition was rendered, checked with a server-side dry run, and exercised on its intended cluster;
- matrix-backed changes pass `python3 scripts/kustomize-matrix.py check`; and
- generated matrix files were regenerated, reviewed, and not hand-edited.

Static rendering, schema, and matrix checks do not prove runtime correctness or performance. Include the relevant target-cluster qualification when requesting review.
