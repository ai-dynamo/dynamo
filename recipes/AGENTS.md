<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- For recipe Kustomize sources, use convention over per-recipe render config:
  `kustomize/base/` is shared input and never renders directly; public overlays
  under `kustomize/overlays/<name>/` render to `deploy-<name>.yaml`; use
  `kustomize/overlays/generic/` for a generic deployable variant; overlays whose
  directory starts with `_` are intermediate and not rendered; shared building
  blocks belong under `kustomize/components/`. Prefer resource-shaped Kustomize
  merge patches over JSON patches where possible. Kustomize bases that patch
  Dynamo CRDs include the central `recipes/kustomize/components/dynamo-openapi/`
  Component; its schema is generated from every operator CRD. The central
  `recipes/kustomize/components/disagg-workers/` Components require one DGD per
  base with backend-neutral `PrefillWorker` and `DecodeWorker` service keys.
  A recipe matrix at `.kustomize-matrix.yaml` has an explicit `source`, a
  `nameTemplate`, and a `matrix` mapping whose values contain a `name` and a
  `components` list. Edit matrix and Component sources, then run
  `scripts/kustomize-matrix.py unfold <matrix.yaml>` followed by
  `scripts/kustomize-matrix.py render <matrix.yaml>`; do not hand-edit generated
  overlay `kustomization.yaml` files, `deploy-*.yaml` files, or the central
  generated schema. `scripts/kustomize-matrix.py check` validates every matrix and
  detects artifacts left by moved matrices; use `unfold --clean` and `render --clean`
  to remove those explicitly.
  To compose additional Components without a checked-in overlay, use
  `scripts/kustomize-matrix.py compose <target> [<component-path>...] [<build-options>...]`.
  The target must be first, Components follow it, and Kustomize build options come
  last.
