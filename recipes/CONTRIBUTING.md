<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recipes Contributing Guide

When adding new model recipes, ensure they follow the standard structure:

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

## Kustomize Variants

Use Kustomize when a recipe has a shared deployment shape plus cloud-provider or
network-provider variants. Recipe-local bases, Components, and generated public
overlays live under `<deployment>/kustomize/`. Shared Components reusable by
multiple recipes live under `recipes/kustomize/components/`. Shared template
sources live under `recipes/kustomize/templates/`. Run the commands in this guide
from the repository root. Keep the checked-in manifests apply-able and easy to
review:

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

Kustomize is both the authoring model and the documentation of a variant: the
base and Components explain individual settings, while each checked-in public
overlay documents the selected composition. The rendered `deploy-<name>.yaml`
is the exact, fully materialized result.

### Using A Variant

Recipe users may apply a checked-in rendered manifest directly:

```bash
kubectl apply -f <deployment>/deploy-<name>.yaml -n ${NAMESPACE}
```

They may instead inspect or apply the checked-in public Kustomization, which
documents the base and selected Components:

```bash
kubectl apply -k <deployment>/kustomize/overlays/<name> -n ${NAMESPACE}
```

Users can also create an uncommitted `kustomization.yaml` in the repository
checkout and apply it with `kubectl apply -k`. For an ad hoc composition without
creating a directory, `compose` creates a temporary Kustomization and writes the
real Kustomize output to stdout. Its target comes first, followed by Components
and then Kustomize build options:

```bash
scripts/kustomize-matrix.py compose \
  <target-kustomization> \
  <component-path>... \
  | kubectl apply -f - -n ${NAMESPACE}
```

None of these user workflows requires `unfold` or `render`.

### Contributing A Variant

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
- Bases that patch Dynamo CRDs include the central
  `recipes/kustomize/components/dynamo-openapi/` Component. Its generated
  schema is derived from every operator CRD and lets strategic merge patches
  merge CRD map lists such as `env` by name.
- The central `recipes/kustomize/components/disagg-workers/` Components apply
  to bases containing one DGD with backend-neutral `PrefillWorker` and
  `DecodeWorker` service keys.

Prefer resource-shaped Kustomize merge patches over JSON patches where possible.
For other Custom Resource Definition (CRD) list fields, include the complete
intended list in the merge patch unless the schema supplies an OpenAPI merge key.

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
`kustomize/overlays/<name>/components/efa/`. A template directory contains
`kustomization.yaml.j2` and may contain a plain `values.yaml` mapping. The
Jinja source must render one Kustomize `Component`.
It receives `values` and an indexed `base` rendered from the matrix `source`.
`base` is indexed by lower-case Kind and `metadata.name`, for example
`base.configmap[values.PREFILL_CONFIG]`. When exactly one resource of a Kind is
expected, use `base.dynamographdeployment | only`; this fails clearly if the
source changes to contain zero or multiple such resources. Templates may use
embedded patches or ordinary local Kustomize source paths. `unfold` materializes
the complete template directory beneath the generated Component, so its local
paths remain valid under Kustomize's load restrictions. It copies every template
file except `values.yaml`; files ending in `*.j2` are rendered without that
suffix. A template can select shared Components; when it does, `unfold` rebases
those external Component paths for its generated location. Jinja rendering uses
strict, immutable sandboxed values: undefined names and attempts to mutate data
are errors.

Regenerate derived artifacts in order: `unfold` writes every checked-in Level-2
public overlay `kustomization.yaml` file for the matrix; `render` invokes
Kustomize and writes every Level-3 `deploy-<name>.yaml` manifest for the matrix
and the central CRD schema:

```bash
scripts/kustomize-matrix.py unfold <matrix.yaml>
scripts/kustomize-matrix.py render <matrix.yaml>
```

To inspect only one concrete public overlay without regenerating the matrix,
run:

```bash
kustomize build <deployment>/kustomize/overlays/<name>
```

For dependent Components, use flat, explicit names such as `aws-efa` and
`vllm-disagg/aws-efa`. An instance-type template can include a generic fabric
Component and patch the hardware-specific resource count derived from the base
deployment.

`render` runs `kustomize build` and falls back to `kubectl kustomize` when
`kustomize` is not on `PATH`. Kustomize drops comments while rendering Kubernetes
objects, so it re-inserts non-SPDX comments from the source YAML before matching
rendered fields. It does not copy comments inside literal block scalars because those
already render in place. It also refreshes the central OpenAPI schema from the
operator CRDs. `scripts/kustomize-matrix.py check` validates all generated overlays,
manifests, and the schema; the Recipe Check CI job runs the same command.
It also reports artifacts left by a moved matrix. Normal generation leaves those
artifacts in place; after reviewing them, clean them explicitly:

```bash
scripts/kustomize-matrix.py unfold --clean <matrix.yaml>
scripts/kustomize-matrix.py render --clean <matrix.yaml>
```

## Validation

The `run.sh` script expects this exact directory structure and will validate that the directories and files exist before deployment:

- Model directory exists in `recipes/<model>/`
- Framework is one of the supported frameworks (vllm, sglang, trtllm)
- Framework directory exists in `recipes/<model>/<framework>/`
- Deployment directory exists in `recipes/<model>/<framework>/<deployment>/`
- Required deploy files exist in the deployment directory (`deploy.yaml` for
  simple recipes, or `deploy-<name>.yaml` for Kustomize variants)
- If present, performance benchmarks (`perf.yaml`) will be automatically executed
