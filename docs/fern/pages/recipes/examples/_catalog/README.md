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
2. One card in `overview.mdx`, mirrored on the matching subsection overview page. Tests require the
   card metadata and destination to stay aligned. Fern resolves the card's native link for each
   version and locale.
3. A page entry under **Recipes → Examples** in `docs/fern/index.yml`.

Each Examples subsection starts with an overview page that uses `ExamplesCatalog` and mirrors the
matching cards from `overview.mdx`. `overview.mdx` remains the catalog source of truth. Keep the
mirrored cards scoped to one `data-topic`; the catalog test checks their metadata and links against
the source cards. Subsection cards omit the topic label, and the page sets
`<ExamplesCatalog topicFilter={false}>`: when every card shares a topic, neither adds information.

`diffusion-overview.mdx` is a specialized topic landing page, not another source-picker example. Its
`DiffusionCatalog` component supplies server-rendered styling and a search/filter/sort widget;
native MDX images and one stretched page link per card preserve Fern's asset, version, and locale
rewriting. The widget reads the existing card metadata, the modality label, and chip labels, without
a second catalog or browser-side metadata requests. Keep the title's `dynamo-diffusion-name` class
so name sorting uses the visible title, and end the top row with the `dynamo-diffusion-modality`
type badge so it sits top right and the type filter uses its text. Start the badge with the
`<Icon>` that `overview.mdx` uses for the destination page, such as `image` or `film`; `data-case`
sets the matching color for the badge and the card's top bar. The title already names the
model, so the Hugging Face ID stays in `data-model`, where search reads it, rather than in the
visible card. Cards carry no description line or arrow; the stretched link keeps its descriptive
accessible name. Do not embed DGDs or add secondary links to these cards. Keep it first under
**Examples → Diffusion** and linked from the main overview, without an `ExampleSelector` or a
duplicate main-catalog card.

Cards identify the source manifest in `data-source` for validation, and carry model, modality,
backend, weight-size, and experimental metadata. Experimental and regular entries share one grid.
Weight-size chips use the pinned Hugging Face file metadata in `diffusion-model-sizes.yaml`: decimal
GB of checkpoint weights, including pipeline components, excluding duplicate root exports. These
are not GPU memory estimates. The chip shows only the value, such as `33.7 GB`; its `title` and the
note below the grid explain the basis. Update the snapshot and visible chip values together. The
snapshot also records the sources of the added provider logos.

The overview cards are the catalog source, matching the model-recipe landing-page pattern. They
are not generated and do not use the model-recipe performance schema. Each card declares:

- `data-example`: the unique page basename, without `.mdx`.
- `data-topic` and `data-topic-label`: the filter key and its display label.
- `data-targets`: space-separated `platform:backend` pairs. Declare actual combinations, not the
  Cartesian product of supported platforms and backends. For example, LoRA includes
  `local:sglang local:vllm kubernetes:vllm`, not `kubernetes:sglang`.
- `data-keywords`: source filenames and any additional search terms.

Inside the card, separate each part with a blank line; without them, MDX merges the parts and Fern
leaves the link unresolved. In order:

1. `<span className="dynamo-example-icon"><Icon icon="…" /></span>`: a Font Awesome icon from
   Fern's built-in `Icon`, tinted with the topic's accent color. Prefer the icon the navigation
   already uses for the concept, such as `arrows-split-up-and-left` for disaggregation or `route`
   for KV-aware routing, and keep icons unique within a topic. Fern renders an unknown icon name
   as an empty tile, so preview new icons.
2. On `overview.mdx` only, `<p className="dynamo-example-topic">` containing the
   `data-topic-label` text.
3. The title, as a plain `###` heading. Do not link it: Fern's heading click handler copies the
   heading anchor instead of following a link inside the heading.
4. A one-line description.
5. `<ExampleTargets targets="…" />` with the same pairs as `data-targets`. It renders one chip per
   backend, with a Kubernetes or Local / CLI glyph for each platform that backend supports.
6. `<a className="dynamo-example-card-link" href="<page>.mdx">Open <title> example</a>`: the card's
   only link, stretched over the whole card.

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
pairs. Search also reads `data-topic-label`, which subsection cards do not display. Cards and links
remain readable without JavaScript. Keep native links in MDX; moving them into component data would
bypass Fern's version/locale rewriting.

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
