<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Authoring Examples

Examples are deployment-first pages, not performance-tuned model recipes. The selector and its
source embeds include only DGD/DGDR manifests and inference launch scripts. Keep each page to a
short introduction, required safety or setup notes, and source-file embeds. Do not copy source-file
contents into Markdown.

Each example has three authoring surfaces:

1. A topic page beside `overview.mdx`, with a `<Code src="…">` for each source asset. Use platform
   and backend choices only for variants that exist. Keep multi-resource DGD files intact.
2. One card in `overview.mdx`. Its Markdown link is resolved by Fern for each version and locale.
3. A page entry under **Recipes → Examples** in `docs/fern/index.yml`.

`diffusion-overview.mdx` is a topic landing page, not another source-picker example. Its
`DiffusionCatalog` component supplies server-rendered styling; native MDX images and one stretched
page link per card preserve Fern's asset, version, and locale rewriting. Do not embed DGDs or add
secondary links to these cards. Keep it first under **Examples → Diffusion** and linked from the
main overview, without an `ExampleSelector` or a duplicate main-catalog card.

Cards identify the source manifest in `data-source` for validation, and carry model, modality,
backend, weight-size, and experimental metadata. Experimental and regular entries share one grid.
Weight-size chips use the pinned Hugging Face file metadata in `diffusion-model-sizes.yaml`: decimal
GB of checkpoint weights, including pipeline components, excluding duplicate root exports. These
are not GPU memory estimates. Update the snapshot and visible chip values together. The snapshot
also records the sources of the added provider logos.

The overview cards are the catalog source, matching the model-recipe landing-page pattern. They
are not generated and do not use the model-recipe performance schema. Each card declares:

- `data-example`: the unique page basename, without `.mdx`.
- `data-topic` and `data-topic-label`: the filter key and its display label.
- `data-targets`: space-separated `platform:backend` pairs. Declare actual combinations, not the
  Cartesian product of supported platforms and backends. For example, LoRA includes
  `local:sglang local:vllm kubernetes:vllm`, not `kubernetes:sglang`.
- `data-keywords`: source filenames and any additional search terms.

Each detail page uses one `ExampleSelector` with a `variants` array. Each entry declares `id`,
`target` (`platform:backend`), `label` (the short option name), and `description`. Match each ID to
one `<div className="dynamo-example-variant" data-example-variant="id">` containing its native
source embeds and any necessary notes. The metadata array drives all the selection rows; do not
repeat platform/backend lists elsewhere in the page.

Platform and backend share one compact control panel. An Example row appears only when the chosen
target has multiple source bundles. One bundle is always selected and server-rendered open; do not
wrap it in Fern tabs or accordions. Source blocks sit directly below the controls, with no extra
card padding. For cloud examples, identify the provider in the option label and keep its setup
notes in the corresponding source bundle.

Set `hide-toc: true` on these source-picker pages. The shared `examples-layout.ts` stylesheet keeps
TOC-less articles aligned with the normal content column without changing Fern's width limits.

`ExamplesCatalog.tsx` discovers the cards, builds topic options, and filters by search text and target
pairs. Cards and links remain readable without JavaScript. Keep native links in MDX; moving them into
component data would bypass Fern's version/locale rewriting.

Supporting assets are deferred from the site catalog for now: Helm values, infrastructure manifests,
clients, setup helpers, standalone engine configurations, chat templates, policy snippets, ECS task
definitions, and Compose files. Leave their original files in `examples/`; do not add them as selector
choices or extra code blocks within a deployment choice. Link existing setup guides for prerequisites.
A ConfigMap, PVC, or other resource bundled inside a DGD file remains part of that source file.

Check manifest contents, not filenames: a DGDR override example is still a complete DGDR, whereas
an engine configuration named `snapshot.yaml` is not a deployment. A DGD-shaped contract fragment
(such as the power-budget example) is also out of scope. Remove an emptied page and its catalog/nav
entries together, preserving its development URL with a redirect.

Use actual backend values. DGDR choices derive their backend from `spec.backend`, not a generic
“Platform configuration” label. Cloud choices here are Kubernetes DGD manifests; keep EKS/GKE in
the example labels rather than mixing cloud services into the platform row.

Single-choice rows still render the same selected button, border, and background as multi-choice
rows. Selecting the already-active platform or backend must not reset the selected example.
Mark experimental deployments explicitly. Presence in this catalog is not proof of runtime
validation or benchmarking.

Validate with:

```bash
python3 -m pytest -c docs/fern/scripts/pytest.ini -p no:cacheprovider --noconftest docs/tests/test_examples_catalog.py
python3 docs/fern/scripts/docs_lint.py --scan docs
cd docs/fern && fern check && fern docs broken-links
```
