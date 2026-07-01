<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- Keep changes focused and reviewable.
- For recipe Kustomize sources, use convention over per-recipe render config:
  `kustomize/base/` is shared input and never renders directly; public overlays
  under `kustomize/overlays/<name>/` render to `deploy-<name>.yaml`; use
  `kustomize/overlays/generic/` for a generic deployable variant; overlays whose
  directory starts with `_` are intermediate and not rendered; shared building
  blocks belong under `kustomize/components/`. Prefer resource-shaped Kustomize
  merge patches over JSON patches where possible. Kustomize bases that patch
  Dynamo CRDs include the central `recipes/kustomize/components/dynamo-openapi/`
  Component; its schema is generated from every operator CRD. Edit the Kustomize
  source, then run `python3 scripts/render_recipe_kustomize.py`; do not hand-edit
  generated `deploy-*.yaml` files or the central generated schema.
- Use Conventional Commit PR titles: `type(scope): summary`. Accepted types:
  `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `perf`, `chore`, `revert`,
  `style`, and `build`.
- PR descriptions must include `Summary` and `Validation`.
- Sign every commit with DCO: `git commit -s`.
