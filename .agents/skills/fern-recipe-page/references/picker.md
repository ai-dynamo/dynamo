<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The target picker

Pure CSS, no JavaScript. Radios drive `body:has(...)` rules in
`docs/fern/components/RecipeStyles.tsx` that show and hide content by
`data-*` attribute. It survives Fern's static render, "View as Markdown" and
`llms.txt`, and degrades to everything-visible on older browsers.

## Row order

Four rows, always in this order. Include a row for each dimension the recipe
has at least one value for; a single-value row is fine and tells the reader
something ("this recipe is vLLM only").

| Row | `name=` | Purpose |
|---|---|---|
| GPU | `recipe-sku` | hardware |
| Workload | `recipe-usecase` | traffic shape |
| Backend | `recipe-engine` | inference engine |
| Topology | `recipe-variant` | aggregated / disaggregated / build variant |

Backend is its own row. Folding it into Topology — variant values like
`trtllm-agg` — was the original source of unreachable states, because it let
a page express combinations no recipe ships.

## Vocabulary

Every value is hardcoded in `RecipeStyles.tsx`. **A value not listed there
renders a control that filters nothing** — the picker looks fine and does
nothing. Check before inventing one:

```bash
grep -o 'name="recipe-[a-z]*"\]\[value="[a-z0-9-]*"' docs/fern/components/RecipeStyles.tsx | sort -u
```

| Dimension | Values |
|---|---|
| `recipe-sku` | `b200` `h200` `h100` `gb200` `gb300` `hopper` `blackwell` |
| `recipe-usecase` | `chat` `agentic` `static` `multimodal` |
| `recipe-engine` | `vllm` `sglang` `trtllm` |
| `recipe-variant` | `agg` `disagg` `disagg-single-node` `disagg-multi-node` `standard` `efa` `kvbm` |

Adding a value means adding its hide rule, its `data-needs-*` rule and its
tab-order companion in `RecipeStyles.tsx`. Do that deliberately; prefer an
existing value where one fits. `hopper` and `blackwell` exist precisely so a
recipe validated across `h100/h200/a100` can offer one chip.

## Conditional options

When a target only exists for some selections, gate its option rather than
letting a reader pick a combination with nothing behind it. Put
`data-needs-*` on the **label**:

```jsx
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg"
       data-needs-sku="b200" data-needs-usecase="agentic">Disaggregated</label>
```

Space-separate to allow several: `data-needs-sku="b200 h200"`. A label with no
`data-needs-*` is always offered, so this is additive — existing pickers are
unaffected.

Work out the minimal set that makes reachable combinations exactly equal the
shipped target set. Over-constraining strands readers; under-constraining puts
back the blank deploy step.

**CSS cannot un-check a radio.** A well-formed set of `data-needs-*` makes bad
states unclickable, because any option that would strand you is itself hidden
by the time you could click it. Hidden options are also removed from the tab
order, so they cannot be keyboard-focused. State restored by the browser is the
only remaining path, which is why the validator reports stranded states
separately.

## Tagging content

Per-target content goes in a wrapper carrying the dimensions that discriminate
it. Omit a dimension that does not:

```jsx
<div data-sku="b200" data-variant="disagg">

```bash
kubectl apply -f recipes/<model>/vllm/disagg-b200/deploy.yaml -n ${NAMESPACE}
```

</div>
```

**MDX rule:** blank line after the opening `<div ...>` and before `</div>`, and
code fences at column 0. Without the blank lines the markdown inside is not
parsed as markdown.

Space-separate to apply to several: `data-sku="b200 h200"`.

## Summary panels

`.dynamo-target-picker-summary` panels carry the selected target's full
configuration, one `<span><b>Label</b> value</span>` per fact. Each panel is
tagged like any other block. Values may contain inline `` `code` `` — the cell
is a block container, so an inline run wraps as prose.

## What not to tag

- **Headings.** A `##` inside a wrapper is invisible for every other selection.
- **Comparison-table rows.** The global rule hides any element with a
  non-matching `data-*`, `<tr>` included, so tagging a results row filters it
  out rather than dimming it. Tables meant to show every target — Performance
  results, Optimization targets — carry no `data-*` on rows.

## Checking

```bash
python3 .agents/skills/fern-recipe-page/scripts/check_recipe_page.py <slug>
```

Enumerates reachable combinations, confirms each renders content in every
`##`/`###` section that has `data-*` blocks, and confirms every catalog deploy
asset is referenced.
