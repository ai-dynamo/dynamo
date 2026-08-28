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

Use this workflow when repository users need multiple maintained public provider or network variants of one deployment shape. Every selected Component must target the base manifest's API and field paths. The current shared provider Components are the legacy alpha strategic-merge path: they target `spec.services` and cannot convert or patch a beta `spec.components` base. The private [beta cluster starter](templates/kustomize/README.md) instead uses guarded JSON 6902 patches against canonical beta component positions and does not include the central OpenAPI Component.

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
- Legacy strategic-merge bases that patch Dynamo CRDs include the central `recipes/kustomize/components/dynamo-openapi/` Component. Its generated schema is derived from every operator CRD and lets strategic merge patches merge CRD map lists such as `env` by name. Guarded JSON 6902 Components, including the beta cluster starter, do not include it.
- The central `recipes/kustomize/components/disagg-workers/` Components apply to bases containing one DGD with backend-neutral `PrefillWorker` and `DecodeWorker` service keys.

Within the legacy alpha matrix path, prefer resource-shaped Kustomize merge patches where possible. For other Custom Resource Definition (CRD) list fields, include the complete intended list in the merge patch unless the schema supplies an OpenAPI merge key. Use guarded JSON 6902 for the beta cluster starter and beta Components whose correctness depends on canonical list positions and fail-loud preconditions.

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
- the portable base omits credential Secret references and relies on operator probes unless a complete, documented per-recipe override is intentional;
- beta components retain the offline settings, backend workers retain the standard security context, and every container uses the catalog's exec form and image-pull policy;
- probe ownership belongs to either the base or the optional `probes` Component, never both;
- the target-cluster composition was rendered, checked with a server-side dry run, and exercised on its intended cluster;
- matrix-backed changes pass `python3 scripts/kustomize-matrix.py check`; and
- generated matrix files were regenerated, reviewed, and not hand-edited.

Static rendering, schema, and matrix checks do not prove runtime correctness or performance. Include the relevant target-cluster qualification when requesting review.
