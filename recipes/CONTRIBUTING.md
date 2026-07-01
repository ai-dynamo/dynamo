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
├── deploy-generic.yaml
├── deploy-aws-efa.yaml
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
        ├── aws-efa/
        │   ├── kustomization.yaml
        │   └── patch-dgd.yaml
        ├── gcp-roce/
        │   ├── kustomization.yaml
        │   └── patch-dgd.yaml
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

Prefer resource-shaped Kustomize merge patches over JSON patches where possible.
For other Custom Resource Definition (CRD) list fields, include the complete
intended list in the merge patch unless the schema supplies an OpenAPI merge key.

Edit the Kustomize source, not the generated manifests. The renderer delegates to
`kustomize build` and falls back to `kubectl kustomize` when `kustomize` is not on
`PATH`. Kustomize drops comments while rendering Kubernetes objects, so the renderer
re-inserts non-SPDX comments from the base and overlay YAML before matching rendered
fields. It does not copy comments inside literal block scalars because those already
render in place. The renderer also refreshes the central OpenAPI schema from the
operator CRDs. Render from the repo root:

```bash
python3 scripts/render_recipe_kustomize.py
```

The pre-commit hook runs the same renderer. If you remove or rename an overlay,
also remove the stale `deploy-<name>.yaml` file that is no longer generated.

## Validation

The `run.sh` script expects this exact directory structure and will validate that the directories and files exist before deployment:

- Model directory exists in `recipes/<model>/`
- Framework is one of the supported frameworks (vllm, sglang, trtllm)
- Framework directory exists in `recipes/<model>/<framework>/`
- Deployment directory exists in `recipes/<model>/<framework>/<deployment>/`
- Required deploy files exist in the deployment directory (`deploy.yaml` for
  simple recipes, or `deploy-<name>.yaml` for Kustomize variants)
- If present, performance benchmarks (`perf.yaml`) will be automatically executed
