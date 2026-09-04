<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- Recipe-local Kustomize sources live in `<deployment>/kustomize/`: `base/` is
  shared input and never renders directly; `components/` holds building blocks
  used only by that recipe; public overlays under `overlays/<name>/` render to
  `deploy-<name>.yaml`; an `overlays/generic/` variant renders when it exists.
  Shared Components selected by multiple recipes live under
  `recipes/kustomize/components/`; that directory has no base or overlays and
  never renders by itself. The copy-and-fill beta cluster scaffold lives under
  `recipes/templates/kustomize/`; it uses guarded JSON 6902 patches against
  canonical `nvidia.com/v1beta1` component positions, requires standalone
  Kustomize v5.8.1, and does not include the central OpenAPI Component. Keep
  filled site values outside the repository. Portable templates omit
  credential Secret references and probe structs. They use the canonical
  `shared-model-cache` bundle, exec-form runtime commands, and
  `imagePullPolicy: IfNotPresent`. Every beta component sets both offline
  environment variables; beta Frontends do not mount the cache, and beta
  backend workers run as UID/GID 0 with `IPC_LOCK`, `SYS_PTRACE`, and
  `SYS_RESOURCE`. The scaffold's optional `probes` Component owns complete
  worker startup-probe overrides. Place it after registry credentials and
  before scheduling. In the legacy alpha matrix path, prefer
  resource-shaped Kustomize merge patches where possible. Legacy bases
  that use strategic merge against Dynamo CRDs include the central
  `recipes/kustomize/components/dynamo-openapi/` Component; its schema is
  generated from every operator CRD. The central
  `recipes/kustomize/components/disagg-workers/` Components require one DGD per
  base with backend-neutral `PrefillWorker` and `DecodeWorker` service keys.
  A recipe matrix at `.kustomize-matrix.yaml` has an explicit `source`, a
  `nameTemplate`, and a `matrix` mapping whose values contain a `name` and may
  provide `components`, `templates`, `values`, and `sortOptions`. The matrix,
  recipe-local base and Components, and shared Components are source.
  `sortOptions` orders the resources in that value's generated overlay. Each
  template selection has a source relative to the matrix and a generated `path`
  under the overlay's `components/` directory. Generated paths
  selected by one variant must be unique and non-overlapping. Shared
  template sources live in `recipes/kustomize/templates/`. A selected template
  directory extends the direct `*.yaml` and `*.yaml.j2` files in its parent
  directory by convention. Only direct files participate; never scan or copy
  subdirectories.
  Parent files are loaded first, then same-output-name files in the selected
  directory replace them (`patch.yaml` and `patch.yaml.j2` both produce
  `patch.yaml`, so either form replaces the other). Optional `values.yaml`
  mappings use shallow parent-to-selected replacement before matrix values. The
  effective files must provide a `kustomization.yaml` or
  `kustomization.yaml.j2` that renders a Component.
  `unfold` evaluates the template with strict sandboxed Jinja, the variant values,
  and the fully rendered base indexed as `base.<lowercase-kind>[metadata.name]`.
  Use `base.<lowercase-kind> | only` only when exactly one such resource is
  required. Templates render one Component. `unfold` materializes the complete
  effective files at its selected path under the generated overlay. Files ending
  in `*.j2` are rendered without that suffix. A template may reference shared
  Components or resources; `unfold` rebases those external paths for the
  generated location.
  Kustomize is both the authoring model and recipe documentation: the base and
  Components explain settings, public overlay `kustomization.yaml` files
  document a concrete variant, and `deploy-*.yaml` files are the fully
  materialized result. Users may apply a checked-in manifest
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
