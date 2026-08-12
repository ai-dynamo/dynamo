<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- Recipe-local Kustomize sources live in `<deployment>/kustomize/`: `base/` is
  shared input and never renders directly; `components/` holds building blocks
  used only by that recipe; public overlays under `overlays/<name>/` render to
  `deploy-<name>.yaml`; an `overlays/generic/` variant renders when it exists.
  Shared Components selected by multiple recipes live under
  `recipes/kustomize/components/`; that directory has no base or overlays and
  never renders by itself. Prefer resource-shaped Kustomize merge patches over
  JSON patches where possible. Bases that patch Dynamo CRDs include the central
  `recipes/kustomize/components/dynamo-openapi/` Component; its schema is
  generated from every operator CRD. The central
  `recipes/kustomize/components/disagg-workers/` Components require one DGD per
  base with backend-neutral `PrefillWorker` and `DecodeWorker` service keys.
  A recipe matrix at `.kustomize-matrix.yaml` has an explicit `source`, a
  `nameTemplate`, and a `matrix` mapping whose values contain a `name` and a
  `components` list. The matrix, recipe-local base and Components, and shared
  Components are source. A dimension value may also select `templates` and set
  `values`. Each template selection has a source relative to the matrix and a
  generated `path` under the overlay's `components/` directory. Shared template
  sources live in `recipes/kustomize/templates/`; a template directory has
  `kustomization.yaml.j2` and optional `values.yaml`.
  `unfold` evaluates the template with strict sandboxed Jinja, the variant values,
  and the fully rendered base indexed as `base.<lowercase-kind>[metadata.name]`.
  Use `base.<lowercase-kind> | only` only when exactly one such resource is
  required. Templates render one Component. `unfold` materializes the complete
  template directory at its selected path under the generated overlay, so its
  local Kustomize paths remain valid; additional `*.j2` files render without
  their suffix. A template may select shared Components; `unfold` rebases those
  external paths for the generated location. Kustomize is both the authoring
  model and recipe documentation: the base and Components explain settings, public overlay
  `kustomization.yaml` files document a concrete variant, and `deploy-*.yaml`
  files are the fully materialized result. Users may apply a checked-in manifest
  with `kubectl apply -f`, a checked-in public overlay with `kubectl apply -k`,
  or an overlay they compose themselves. Contributors run all matrix commands
  from the repository root. `scripts/kustomize-matrix.py unfold <matrix.yaml>`
  writes every generated public overlay and selected generated Component for that
  matrix; follow it with
  `scripts/kustomize-matrix.py render <matrix.yaml>` to regenerate every
  checked-in manifest for that matrix and the central schema. To inspect only
  one concrete overlay, run `kustomize build <deployment>/kustomize/overlays/<name>`.
  Do not hand-edit generated artifacts. `scripts/kustomize-matrix.py check`
  validates every matrix and detects artifacts left by moved matrices; use
  `unfold --clean` and `render --clean` to remove those explicitly. To compose
  additional Components without a checked-in overlay, use
  `scripts/kustomize-matrix.py compose <target> [<component-path>...] [<build-options>...]`.
  The target must be first, Components follow it, and Kustomize build options
  come last.
