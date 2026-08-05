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
network-provider variants. Keep the checked-in manifests apply-able and easy to
review:

```text
<deployment>/
├── .kustomize-matrix.yaml
├── deploy-generic.yaml
├── deploy-aws-efa-p8d16.yaml
├── deploy-gcp-roce.yaml
├── perf.yaml
└── kustomize/
    ├── base/
    │   ├── deploy.yaml
    │   └── kustomization.yaml
    ├── components/
    │   └── <shared-building-block>/
    └── overlays/
        ├── generic/
        │   └── kustomization.yaml
        ├── aws-efa-p8d16/
        │   └── kustomization.yaml
        ├── gcp-roce/
        │   └── kustomization.yaml
        └── _shared-overlay/
```

The render convention is:

- `kustomize/base/` is shared input and is not rendered directly.
- `kustomize/overlays/<name>/` renders to `deploy-<name>.yaml`.
- `kustomize/overlays/generic/` renders to `deploy-generic.yaml`. Use it when a
  generic deployable variant exists.
- `kustomize/overlays/_<name>/` is intermediate and is not rendered.
- `kustomize/components/` is for shared Kustomize building blocks and is not rendered.
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

Edit the Kustomize source, not the generated manifests. A recipe matrix is an explicit
`.kustomize-matrix.yaml` beside the recipe. It names the Kustomize `source`, a
`nameTemplate`, and matrix dimensions. Every dimension value has a human-readable
`name` and a list of Kustomize `components`; output names interpolate only the value
names, never their paths:

```yaml
source: kustomize/overlays/_rdma
nameTemplate: "${variant}"
matrix:
  variant:
    - name: aws-efa-p8d16
      components:
        - ../../../kustomize/components/aws/components/efa/components/p8d16
```

Generate the checked-in Level-2 overlays, then the flattened Level-3 manifests:

```bash
scripts/kustomize-matrix.py unfold <matrix.yaml>
scripts/kustomize-matrix.py render <matrix.yaml>
```

For a dependent Component tree, select only the leaf in the matrix. Kustomize
rejects a Component that includes an ancestor directory as a load cycle, so keep
each parent implementation in a sibling `_component/` directory. The leaf includes
that sibling, rather than its containing directory.

`render` runs `kustomize build` and falls back to `kubectl kustomize` when
`kustomize` is not on `PATH`. Kustomize drops comments while rendering Kubernetes
objects, so it re-inserts non-SPDX comments from the source YAML before matching
rendered fields. It does not copy comments inside literal block scalars because those
already render in place. It also refreshes the central OpenAPI schema from the
operator CRDs. `scripts/kustomize-matrix.py check` validates all generated overlays,
manifests, and the schema; the pre-commit hook runs the same command.
It also reports artifacts left by a moved matrix. Normal generation leaves those
artifacts in place; after reviewing them, clean them explicitly:

```bash
scripts/kustomize-matrix.py unfold --clean <matrix.yaml>
scripts/kustomize-matrix.py render --clean <matrix.yaml>
```

For an ad-hoc, uncommitted composition, use `compose`. The target is first,
Components follow it, and Kustomize build options come last:

```bash
scripts/kustomize-matrix.py compose \
  <target-kustomization> \
  /absolute/path/to/recipes/kustomize/components/aws/components/efa/components/p8d16 \
  --enable-helm
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
