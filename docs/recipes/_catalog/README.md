<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Authoring Recipe & Feature Benchmark pages

This guide covers the **page side** of the Recipes and Feature Benchmarks
surfaces — the MDX structure, the target-picker component, and the
CSS-coupled vocabulary. For the machine-readable catalog contract, see the
schema header in [`recipes.yaml`](recipes.yaml) and
[`benchmarks.yaml`](../../benchmarks/_catalog/benchmarks.yaml).

> [!IMPORTANT]
> The picker is **pure CSS** (no JavaScript). Every allowed dimension value is
> hardcoded in `fern/main.css`. A page that uses a value not listed below will
> render the picker but **filter nothing** — adding a new value requires a
> `fern/main.css` edit (see [Adding a new axis value](#adding-a-new-axis-value)).

## Steps to add a page

1. **Write the MDX** at `docs/recipes/<slug>.mdx` (or `docs/benchmarks/<slug>.mdx`),
   following the [page blueprint](#page-blueprint) below.
2. **Add a catalog entry** to `docs/recipes/_catalog/recipes.yaml` (or
   `benchmarks.yaml`) per its schema header — every deploy/perf asset path must
   resolve in the repo.
3. **Wire navigation** in `docs/index.yml` (add the `- page:` under the Recipes
   tab or the Feature Benchmarks section).
4. **Patch `fern/main.css` only if** the page introduces a picker axis value not
   already supported (see below).
5. Add the landing-page card (`docs/recipes/README.mdx`) and update the model /
   target counts.
6. Run `fern check` and `fern docs broken-links`; preview with `fern docs dev`.

## Page blueprint

No hero card. Pages open with:

1. Frontmatter: SPDX header, `title`, one-sentence `subtitle`.
2. One short intro paragraph (what it deploys + the strongest proof point).
3. The **target picker** (multi-target) or a **static summary panel**
   (single-target), see below.
4. `## Prerequisites` → `## Deploy` → `## Smoke Test` → `## Benchmark` →
   `## Expected Performance` (omit if no numbers) → `## Compare All Targets`
   (multi-target only) → `## Related Feature Benchmarks` → `## Notes` →
   `## Source`.

Per-variant content is wrapped in `<div data-sku="..." data-usecase="..."
data-variant="...">` blocks; only the selected combination is shown. **MDX
rule:** leave a blank line after `<div ...>` and before `</div>`, and keep code
fences at column 0.

## The target picker

A multi-target page renders one picker with one row per dimension:

```jsx
<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" defaultChecked />
<label htmlFor="recipe-sku-b200">B200 <span className="dynamo-target-picker-hint">Recommended</span></label>
<input type="radio" id="recipe-sku-h200" name="recipe-sku" value="h200" />
<label htmlFor="recipe-sku-h200">H200</label>
</div>
<!-- one .dynamo-target-picker-summary per combination, tagged with its data-* attrs -->
<div className="dynamo-target-picker-summary" data-sku="b200" data-usecase="chat">
<span><b>Checkpoint</b> ...</span>
...
</div>
</div>
```

Then tag every variant-scoped section (`Prerequisites`, `Deploy`, `Smoke Test`,
`Benchmark`, and the `<tr>` rows of the Expected Performance table) with the
matching `data-*` attributes. Space-separated values mean "applies to several"
(e.g. `data-sku="b200 h200"`).

**Single-target pages** use the static form — same panel, no radios, no
`data-*` blocks anywhere:

```jsx
<div className="dynamo-target-picker static">
<p className="dynamo-target-picker-title">Deployment target</p>
<div className="dynamo-target-picker-summary">
<span><b>Checkpoint</b> ...</span>
...
</div>
</div>
```

## CSS-coupled vocabulary

The hide/highlight rules in `fern/main.css` enumerate every allowed value.
A new value renders but filters nothing until CSS is added.

| Dimension (`name=`) | Supported `value=` |
| --- | --- |
| `recipe-sku` | `b200`, `h200`, `h100`, `gb200`, `hopper`, `blackwell` |
| `recipe-usecase` | `chat`, `agentic` |
| `recipe-variant` | `agg`, `disagg`, `disagg-single-node`, `disagg-multi-node`, `trtllm-agg`, `trtllm-disagg`, `vllm-disagg`, `standard`, `efa`, `kvbm` |

Landing-page filter chips (`docs/recipes/README.mdx`) are a separate
vocabulary — `provider`, `runtime`, `hardware`, `technique`, `workload` — also
hardcoded in CSS; a chip that matches no card is a dead end, so only add a chip
when a card carries the matching `data-*` token.

## Adding a new axis value

To support, e.g., an `mi300` SKU or a `disagg-3node` topology:

1. In `fern/main.css`, find the picker `body:has(...)` block and add the
   hide rule:
   ```css
   body:has(input[name="recipe-sku"][value="mi300"]:checked) [data-sku]:not([data-sku~="mi300"]) { display: none; }
   ```
2. If the value participates in Expected-Performance row highlighting, add it to
   the matching `tr[data-...]` rule group too.
3. Then use the value in the page's picker inputs and `data-*` blocks.

Skipping step 1 produces a picker that renders but silently filters nothing.

## Why it's pure CSS

The picker uses `:has()` + sibling/attribute selectors with no JS, so it works
in Fern's static render, survives "View as Markdown" / `llms.txt`, and degrades
safely on older browsers (all variants show stacked rather than breaking).
Hidden variants use `display: none` (clean a11y tree); comparison tables dim
non-selected rows with `opacity` so every row stays readable.

This component styling lives entirely under the `dynamo-*` namespace and only
renders when `fern/main.css` is applied — see the preview-vs-production note in
`.github/workflows/fern-docs.yml` (PR previews skip the global theme so project
CSS renders).
