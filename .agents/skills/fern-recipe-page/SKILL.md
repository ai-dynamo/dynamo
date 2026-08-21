---
name: fern-recipe-page
description: Publish the Fern docs page for a newly released Dynamo recipe, wire it into the Recipes landing page, catalog and navigation, and validate it before review. Use when a recipe lands under recipes/ and needs customer-facing docs, or when an existing recipe page has drifted from its README. Builds the page from the recipe's own READMEs (including perf/, efa/ and per-precision sub-READMEs) as the source of truth, mirrors their heading structure, and assembles the GPU/Workload/Backend/Topology target picker so it offers exactly the targets that ship. Ships check_recipe_page.py, which gates the PR on MDX structure, CSS-coupled picker vocabulary, dead-end selections, path resolution, catalog coverage and landing-page counts, and emits a checklist for the PR description.
license: Apache-2.0
metadata:
  author: Shwetha Krishnamurthy <skrishnamurt@nvidia.com>
  tags:
    - dynamo
    - docs
    - fern
    - recipes
  permissions:
    - file_read
    - file_write
---

# Fern Recipe Page

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Publish the customer-facing Fern page for a recipe, from the recipe's own
READMEs, and prove it is correct before asking for review.

**The recipe README on GitHub is the source of truth.** The Fern page presents
the same information with a target picker on top. It does not add facts the
READMEs do not carry, and it does not contradict them.

## When to use

- A new recipe landed under `recipes/<model>/` and needs a docs page.
- An existing page has drifted from its README.
- A recipe gained or lost a target and the picker needs to match.

## Inputs

The recipe directory, and **every** README inside it — not just the top-level
one. Find them first; sub-READMEs carry the benchmark procedure and the
per-precision detail, and missing them is the most common cause of a thin page:

```bash
find recipes/<model> -name README.md
```

For `deepseek-v4` that is five files; for `qwen3.5-122b`, five; for `qwen3-32b`,
three. A page built from the top-level README alone will be missing the trace
staging step, and its benchmark section will not work.

## Steps

### 1. Read the recipe

Read every README end to end, then the shipped `deploy.yaml` and `perf.yaml`.
Build the target list: for each target, its GPU, workload, backend, topology,
deploy asset, and the values a reader must set.

Where a README contradicts its own manifest — a service name no manifest
defines, an image tag nothing pins — **the manifest wins on the page**, and the
README bug goes in the PR description for its owner. Copying a command that
fails helps nobody.

### 2. Write the catalog entry

Create `docs/fern/pages/recipes/_catalog/recipes/<id>.yaml` with one entry per
shipped target, and add `<id>` to `index.yaml` in nav order. See
`docs/fern/pages/recipes/_catalog/README.md` for the schema. The catalog is the
machine-readable contract the validator checks the page against, so get the
target list right here first.

```bash
python3 docs/fern/pages/recipes/_catalog/validate.py
```

### 3. Write the page

`docs/fern/pages/recipes/model-recipes/<slug>.mdx`.

**Mirror the README's own heading structure and order.** If its sections are
`Configurations / Supported features / Prerequisites / Quick Start /
Configuration notes / Known issues`, use those, with that wording and
capitalisation. Do not impose a house structure the README does not use — a
reader moving between GitHub and the docs should recognise the same page.

Add only `## Source` at the end, linking back to the README and the manifests.

See [references/page-anatomy.md](references/page-anatomy.md) for what belongs to
the README and what is Fern scaffolding.

### 4. Build the target picker

Four rows, always in this order, one row per dimension the recipe has values
for:

```
GPU  ->  Workload  ->  Backend  ->  Topology
```

Gate options that only exist for some selections with `data-needs-*`, so the
picker never offers a combination that ships no recipe. Full contract, the
CSS-coupled vocabulary, and the traps in
[references/picker.md](references/picker.md). Read it before writing the
picker — every value is hardcoded in `RecipeStyles.tsx`, and an unlisted value
renders a control that silently filters nothing.

### 5. Wire it up

- **Landing page** — add a card to `docs/fern/pages/recipes/model-recipes/overview.mdx`
  in catalog order, and update both header counts: model families, and
  deployable configurations (the sum of catalog targets across carded recipes).
- **Navigation** — add a `- page:` entry under the Recipes tab in
  `docs/fern/index.yml`.

### 6. Validate

```bash
python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py <slug>
```

It exits non-zero on failure. Checks MDX structure, picker vocabulary,
dead-end selections, path resolution, catalog target coverage, landing-page
card and counts, and navigation.

Then the repo's own gates:

```bash
python3 docs/fern/pages/recipes/_catalog/validate.py
python3 docs/fern/scripts/check_asset_paths.py
pre-commit run --files docs/fern/pages/recipes/model-recipes/<slug>.mdx
```

If the Fern CLI is available, also `fern check` and `fern docs broken-links`.

### 7. Attach to the PR

```bash
python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py <slug> --pr
```

Paste the block into the PR description under `Validation`. PR descriptions
need `Summary` and `Validation` sections, and commits need `git commit -s`
(see the PR conventions in `AGENTS.md`). State plainly anything you could not
run rather than implying it passed.

## Traps

Each of these has shipped broken at least once.

- **A `##` heading inside a `data-*` wrapper** is invisible for every other
  selection. Walk the file tracking `<div>` depth and confirm every heading sits
  at depth 0. Headings are page scaffolding, never per-target content.
- **The global CSS hides any element with a non-matching `data-*`, including
  `<tr>`.** Tagging a results-table row filters it out rather than dimming it.
  Comparison tables that should show every target must carry no `data-*` on
  their rows.
- **A picker chip can group several catalog targets.** One `hopper` chip may
  cover `h100/h200/a100`, so selectable combinations and catalog target counts
  legitimately differ. The reliable check is that every catalog deploy asset is
  referenced somewhere on the page.
- **A section can look covered while its deploy step is blank.** Check coverage
  per `##` *and* `###` section, not per page: a page-wide `<div data-sku="...">`
  in Prerequisites makes every combination look served.
- **Curl and AIPerf commands often live in the manifests, not the README
  prose.** Do not delete a Smoke Test or Benchmark section just because the
  README has no equivalent heading.
- **`recipes/README.md` is owned elsewhere.** Do not edit it from a docs PR.
