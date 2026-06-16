<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Authoring Recipe & Feature Benchmark pages

This guide covers the **page side** of the Recipes and Feature Benchmarks
surfaces — the MDX structure, the target-picker component, and the
CSS-coupled vocabulary. For the machine-readable catalog contract, see the
[split catalog](#the-machine-readable-catalog) section below.

<Info>
The picker is **pure CSS** (no JavaScript). Every allowed dimension value is
hardcoded in `fern/main.css`. A page that uses a value not listed below will
render the picker but **filter nothing** — adding a new value requires a
`fern/main.css` edit (see [Adding a new axis value](#adding-a-new-axis-value)).
</Info>

## Steps to add a page

1. **Write the MDX** at `docs/recipes/<slug>.mdx` (or `docs/benchmarks/<slug>.mdx`),
   following the [page blueprint](#page-blueprint) below.
2. **Add a catalog entry** — create one per-entry file at
   `docs/recipes/_catalog/recipes/<id>.yaml` (or
   `docs/benchmarks/_catalog/benchmarks/<id>.yaml`) and add its `<id>` to the
   matching `index.yaml`. Each file holds exactly one object that validates
   against `schema.json`; every deploy/perf asset path must resolve in the repo.
   See [the machine-readable catalog](#the-machine-readable-catalog) below, then
   run `python3 docs/recipes/_catalog/validate.py`.
3. **Wire navigation** in `docs/index.yml` (add the `- page:` under the Recipes
   tab or the Feature Benchmarks section).
4. **Patch `fern/main.css` only if** the page introduces a picker axis value not
   already supported (see below).
5. Add the landing-page card (`docs/recipes/README.mdx`) and update the model /
   target counts.
6. Run `fern check` and `fern docs broken-links`; preview with `fern docs dev`.

## The machine-readable catalog

The catalog is the stable, machine-readable contract for what recipes and
benchmarks exist, how to deploy and benchmark them, how they were validated,
and what performance to expect. It is consumed by humans, validation
automation, and agentic harnesses.

It is **split into one file per entry** plus an ordering index (this replaces
the former monolithic `recipes.yaml` / `benchmarks.yaml`, which reduced merge
conflicts and gave a cleaner per-entry ownership/validation path):

```
docs/recipes/_catalog/
  index.yaml            # ordering + inclusion only (recipes + deferred_recipes)
  schema.json           # JSON Schema (draft 2020-12) for ONE recipe object
  recipes/<id>.yaml     # one recipe object per file (active AND deferred)
  validate.py           # catalog validator (covers BOTH catalogs)
docs/benchmarks/_catalog/
  index.yaml            # ordering + inclusion only (benchmarks)
  schema.json           # JSON Schema for ONE benchmark object
  benchmarks/<id>.yaml  # one benchmark object per file
```

- **`index.yaml`** controls sidebar/landing order. The `recipes:` list is the
  active (customer-visible) recipes in nav order; `deferred_recipes:` lists
  recipes held back from the rendered surface (no rendered page). Active order
  matches the Recipes tab in `docs/index.yml`; benchmark order matches the
  Feature Benchmarks section. Every id must have a matching `<id>.yaml` file and
  vice-versa.
- **`<id>.yaml`** holds exactly one entry object as a top-level document, with
  an SPDX header. The internal `id:` must equal the filename. Deferred recipes
  carry a `deferred_reason` and omit `page`; active recipes carry `page`.
- **`schema.json`** is the JSON Schema (draft 2020-12) for a single entry
  object. It does structural validation (required fields, enums, types) and is
  intentionally permissive on nested target/arm values.

### Validating the catalog

Run from the repo root:

```bash
python3 docs/recipes/_catalog/validate.py
```

(The single script validates **both** catalogs.) It checks: index ↔ file
correspondence (no orphans, no dangling entries), internal-`id`/filename match,
no duplicate ids, schema conformance, that every `page:` resolves under `docs/`,
that every deploy/perf/benchmark asset path resolves in the repo, and
cross-catalog referential integrity (recipe `related_benchmarks` ↔ benchmark
ids; benchmark `related_recipes` and `promotion_candidate.deferred_recipe_id` ↔
recipe ids, including deferred). It exits non-zero on any failure.

The validator uses **stdlib only** and degrades gracefully: it prefers `pyyaml`
+ `jsonschema` if importable, otherwise parses YAML with a built-in minimal
parser and falls back to a required-keys check. It prints which mode it ran in.
For full schema validation, `pip install pyyaml jsonschema`.

<Note>
CI does not yet run this validator. **Follow-up:** wire
`python3 docs/recipes/_catalog/validate.py` into the docs CI job so catalog
changes are gated on it.
</Note>

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
