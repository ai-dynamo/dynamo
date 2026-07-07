---
name: dynamo-docs
description: Add, update, move, or remove content on the Dynamo Fern docs site — standard docs pages, catalog-driven recipe and feature-benchmark pages, examples, recipes, and translations — keeping everything in line with the documentation style guide. Use for any change under docs/, recipes/, or examples/ (new page, edit, section move, rename, removal, recipe/benchmark page, .zh-CN translation, version cut) and whenever content needs its frontmatter, headings, links, callouts, or terminology fixed.
---

# Dynamo Docs Maintenance

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Unified skill for adding, updating, moving, and removing content on the Dynamo Fern documentation
site, in line with the project's authoring guides.

Two authoring guides govern this work; read whichever applies before writing:

- [`docs/documentation-style-guide.md`](https://github.com/ai-dynamo/dynamo/blob/main/docs/documentation-style-guide.md) — the standard for **every** page: frontmatter, headings, prose, terminology, links, callouts. The must-fix subset is distilled in [Style Guide Is the Standard](#style-guide-is-the-standard) and [Content Rules](#content-rules) below.
- [`docs/recipes/_catalog/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/docs/recipes/_catalog/README.md) — the standard for **recipe and feature-benchmark pages** (the catalog contract, the `.mdx` page blueprint, and the pure-CSS target picker). See [Add a Recipe or Feature Benchmark Page](#add-a-recipe-or-feature-benchmark-page).

## Branch Rule

**ALL edits happen on `main` (or a feature branch based on `main`).**
The `docs-website` branch is CI-managed and must **never** be edited by hand.

## Style Guide Is the Standard

Every page under `docs/` (and the READMEs under `examples/` and `recipes/`) follows the
[Documentation Style Guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/documentation-style-guide.md)
(`docs/documentation-style-guide.md`). Read it before writing content. The docs bot enforces a
**must-fix** subset on every PR — get these right or the checks fail:

- **SPDX header** on every file, copyright range `2025-2026`. Fern pages put the two `#` lines
  *inside* the `---` frontmatter; plain READMEs use an HTML-comment block.
- **Frontmatter with at least one metadata key** (`title`/`subtitle`/`sidebar-title`) and **no body
  `# H1`**. Fern renders the page H1 from the nav `page:` value, so a body `# H1` produces a
  duplicate title — and a bare `#` SPDX line left in the body also renders as an H1. Start the body
  at `##`.
- **A nav entry** in `docs/index.yml` for every new page — a page not in the nav is unreachable.
- **Links**: relative path *with extension* within `docs/` (`[Routing](router-concepts.md)`);
  absolute `https://github.com/ai-dynamo/dynamo/blob/main/<path>` URL for targets outside `docs/`
  (examples, recipes, source; `/tree/main/` for a directory). No `../` path that escapes `docs/`, and
  never a hardcoded `https://docs.nvidia.com/...` link to a page in this repo. Link text names the
  destination, never "click here".
- **No internal or sensitive references**: NVBug/JIRA/Linear IDs, internal hostnames, secrets,
  `TODO`/`FIXME`.

Everything else in the style guide (page types, heading case, terminology, list and code-fence
formatting, the pre-merge checklist) is guidance — the high-value rules are distilled in
[Content Rules](#content-rules) below; apply them and deviate only with a reason.

## Content Rules

Apply these on every page so the result reads like a person wrote it and passes review without a
round-trip to the style guide. These are defaults; deviate with a reason.

- **Page type (Diátaxis).** Each page serves one need — *tutorial* (`getting-started/`), *how-to*
  (`backends/<engine>/`, `kubernetes/`), *reference* (flags/APIs/config), or *explanation*
  (`design-docs/`). Don't blend a how-to into a flag reference; split and cross-link.
- **Headings.** Title Case for short label / noun-phrase headings ("Routing Behavior"); sentence
  case for full-phrase headings ("Choosing a checkpoint flow"). Be consistent within a page. No end
  punctuation. Logical `##` → `###` hierarchy, no skipped levels. Renaming a heading breaks inbound
  `#anchor` links — rename deliberately.
- **Terminology, exact casing.** Backends: **vLLM**, **SGLang**, **TensorRT-LLM** (or **TRT-LLM**) —
  never "vllm", "Sglang", "TensorRT LLM". **NVIDIA Dynamo** on first mention, then **Dynamo**; **KV
  router**, **NIXL**, **GPU**; **Kubernetes**, not "k8s", in prose. Expand acronyms on first use
  ("Time To First Token (TTFT)"). Use one word per concept.
- **Inclusive terms.** "denylist"/"allowlist", not "blacklist"/"whitelist"; "primary"/"replica", not
  "master"/"slave".
- **Cut marketing and bombast.** Remove "seamless, robust, powerful, blazing-fast, cutting-edge,
  effortless, unlock, leverage, delve, comprehensive, rich ecosystem, world-class, game-changing".
  Cut filler ("it's important to note", "simply", "just", "in order to") and difficulty words
  ("easy", "easily"). Start sentences with a verb; active voice; present tense; second-person
  imperative. Name the flag/default/command, not "configure the appropriate settings". Avoid the
  em-dash-aside tic.
- **Procedures.** Condition before instruction ("To enable KV-aware routing, set `--router-mode
  kv`", not the reverse). One action per numbered step.
- **Links.** Follow the must-fix Links rule in
  [Style Guide Is the Standard](#style-guide-is-the-standard) (relative + extension inside `docs/`,
  absolute GitHub URL outside, no `../` escape, no `docs.nvidia.com` self-link).
- **Code fences** always tag a language (`bash`, not `sh`); no `$`/`#` prompt prefixes; put output in
  its own `text` block. Wrap flags, paths, and `DYN_*` env vars in backticks in prose.
- **Lifecycle.** Mark preview features **Experimental.** and legacy ones **Deprecated.** (with a
  `> [!WARNING]`); note availability for new features ("Available since v0.X").

## Operations

Pick your operation:

- Standard `.md` doc page → [Add a Page](#add-a-page)
- Rendered recipe / feature-benchmark page (`.mdx` + catalog triple) → [Add a Recipe or Feature Benchmark Page](#add-a-recipe-or-feature-benchmark-page)
- Code under `examples/` or `recipes/` → [Add an Example or Recipe (code)](#add-an-example-or-recipe-code)
- Edit, move, or remove existing content → [Update a Page](#update-a-page), [Remove a Page](#remove-a-page) (recipes: [Move, defer, or remove a recipe](#move-defer-or-remove-a-recipe))
- Chinese translation or version cut → [Translations and Versioned Navs](#translations-and-versioned-navs)

### Add a Page

1. **Choose placement from the live nav.** Open `docs/index.yml` and find the existing page closest
   in topic to yours — your page joins **that** section, and its file goes in that sibling's
   subdirectory under `docs/`. Page *type* narrows the field (tutorial → `getting-started/`, how-to →
   `backends/<engine>/` or `kubernetes/`, reference → flags/APIs/config, explanation →
   `design-docs/`), but the nearest existing page is the tie-breaker — don't guess from the section
   names in [Navigation](#navigation-tabs-and-sections), read the file. Note the section, the
   subdirectory, a kebab-case `.md` filename, and the page title.
2. Create `docs/<subdirectory>/<filename>.md`. Frontmatter carries the SPDX header plus at least one
   metadata key; the body starts at `##` with a short intro — **no body `# H1`**:

```markdown
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: <Page Title>
subtitle: <One-line description of the page>
---

Short intro paragraph stating what the page covers.

## <First section>
```

3. Add a nav entry in `docs/index.yml` under the section you chose in step 1 — a `- page:` in that
   section's `contents:`, 2-space indent, `path:` relative to `docs/` (see
   [Navigation](#navigation-tabs-and-sections) for the grammar):

```yaml
- page: <Page Title>
  path: <subdirectory>/<filename>.md
```

### Update a Page

1. Locate by file path, page title, or keyword search (`grep -rn` in `docs/`).
2. **Content only** -- edit the markdown file directly; keep it within the style guide.
3. **Title/label change** -- update the frontmatter (`title`/`sidebar-title`) and the `- page:` name
   in `docs/index.yml`.
4. **Section move** -- `git mv` the file when the subdirectory changes, move the nav entry to the new
   section, and update every incoming link.

> [!IMPORTANT]
> A page's URL is `<section-slug>/<page-name-slug>`, where the page-name slug comes from the nav
> `page:` label. Moving a page to another section **or** renaming its label changes that URL. Add a
> **dev-scoped** redirect to the `redirects:` list in `fern/docs.yml`: `/dynamo/dev/<old>` →
> `/dynamo/dev/<new>`. Editing `docs/index.yml` regenerates only the `dev` nav, so do **not** redirect
> the unversioned (`/dynamo/<old>`) or `/dynamo/latest/<old>` forms — those serve **Latest**, a frozen
> release snapshot that `main` edits don't touch, and a redirect there would break a working URL. See
> [Redirects and the version model](#redirects-and-the-version-model).

### Remove a Page

1. Find incoming links: `grep -rn "<filename>" docs/`.
2. `git rm docs/<subdirectory>/<filename>.md`.
3. Remove the `- page:` block from `docs/index.yml`. If it was the last page in a section, remove the
   whole `- section:` block.
4. Fix or remove every incoming link found in step 1, and add a `fern/docs.yml` redirect if the page
   had a stable URL.

### Add a Recipe or Feature Benchmark Page

Recipe and feature-benchmark pages are **catalog-driven** and use `.mdx` (they embed a pure-CSS
target picker). Authoritative guide:
[`docs/recipes/_catalog/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/docs/recipes/_catalog/README.md).
Each page is a triple — page + catalog entry + nav:

1. **Write the `.mdx`** at `docs/recipes/<slug>.mdx` (or `docs/benchmarks/<slug>.mdx`). Frontmatter
   carries SPDX + `title` + one-sentence `subtitle`; body starts with a short intro, then the target
   picker — multi-target pages use the radio picker, single-target pages use the **static** form
   (exact classes under [Target picker](#target-picker) below) — then the fixed section order:
   `## Prerequisites` → `## Deploy` → `## Smoke Test` → `## Benchmark` → `## Expected Performance`
   (omit if no numbers) → `## Compare All Targets` (multi-target only) → `## Related Feature
   Benchmarks` → `## Notes` → `## Source`. **MDX rule:** blank line after `<div ...>` and before
   `</div>`; keep code fences at column 0.
2. **Add a catalog entry** — one file at `docs/recipes/_catalog/recipes/<id>.yaml` (or
   `docs/benchmarks/_catalog/benchmarks/<id>.yaml`), SPDX header, exactly one object. **Read the
   sibling `schema.json` first for the exact field set** (`docs/recipes/_catalog/schema.json` for
   recipes, `docs/benchmarks/_catalog/schema.json` for benchmarks — they are **different** schemas) —
   each is `additionalProperties: false`, so an invented or misspelled key fails validation; don't
   guess the shape. A **recipe** entry requires `id`,
   `title`, `provider`, `model`, `status`, `targets`, `maintainer`, and each `targets[]` item
   requires `id`, `recommended`, `hardware`, `runtime`, `topology`, `techniques`, `workload`,
   `deploy`, `expected_performance`. Internal `id:` **must equal the filename**; active entries carry
   `page:`, deferred ones carry `deferred_reason` and omit `page:`. Add the `<id>` to the matching
   `_catalog/index.yaml` (`recipes:` for active, `deferred_recipes:` for deferred — it controls
   sidebar/landing order).
3. **Wire navigation** in `docs/index.yml`: a `- page:` under `- tab: recipes` for recipes, or under
   the **Feature Benchmarks** section (`- tab: docs`) for benchmarks. Per-benchmark pages are usually
   `hidden: true` (surfaced from the landing page).
4. **Patch `fern/main.css` only if** the page introduces a picker axis value not already supported
   (`recipe-sku`: `b200`/`h200`/`h100`/`gb200`/`hopper`/`blackwell`; `recipe-usecase`:
   `chat`/`agentic`; `recipe-variant`: `agg`/`disagg`/…). A value missing from CSS renders but
   filters nothing.
5. **Add the landing card** in `docs/recipes/README.mdx` and update the model/target counts.
6. **Validate**: `python3 docs/recipes/_catalog/validate.py` (covers both catalogs), then `fern
   check` and `fern docs broken-links`.

#### Catalog entry shape

`schema.json` is authoritative for the field set; this skeleton just anchors the **nested shapes and
enums** that are easy to get wrong (`model`/`hardware`/`runtime`/`workload`/`deploy`/
`expected_performance` are **objects**, not scalars; `status` and `topology` are **enums**). Minimal
valid active entry:

```yaml
id: llama-3-1-8b                  # == filename; pattern ^[a-z0-9][a-z0-9-]*$
title: Llama 3.1 8B
provider: meta                    # landing-page filter key (meta, qwen, nvidia, …)
model:
  name: Llama 3.1 8B
  hf_id: Meta-Llama/Llama-3.1-8B
  precision: BF16
status: validated                 # enum: validated | experimental  (NOT "active")
page: recipes/llama-3-1-8b.mdx    # active only; deferred → omit page:, add deferred_reason:
maintainer: Jane Doe              # or null (null is tracked as a gap)
targets:                          # >= 1 item
  - id: vllm-agg-h100
    recommended: true             # bool
    hardware: { gpu: H100, count: 1 }
    runtime: { framework: vllm }
    topology: aggregated          # enum: aggregated | disaggregated
    techniques: [bf16]
    workload: { type: chat }
    deploy: { asset: recipes/llama-3-1-8b/vllm/agg/deploy.yaml }
    expected_performance: { available: false }   # add summary: when numbers exist
```

**Benchmarks use a different schema.** A `docs/benchmarks/_catalog/benchmarks/<id>.yaml` entry
validates against `docs/benchmarks/_catalog/schema.json`, whose required set is `id`, `title`, `page`,
`claim`, `subtype` (enum: `ab-test`/`feature-stack`/`topology`/`provider-comparison`/`hands-on`),
`features`, `model`, `hardware`, `traffic`, `arms`, `results`, `maintainer` — **no** `provider`,
`status`, or `targets`. The skeleton above is recipe-only; read the benchmark schema for that shape.

#### Target picker

The picker is pure CSS under the `dynamo-*` namespace — **MDX uses `className`, not `class`**, and the
exact class names matter (a wrong class name, or a `class=`-spelled wrapper, renders but filters nothing). A
**multi-target** page renders `<div className="dynamo-target-picker">` containing a
`dynamo-target-picker-title`, one `dynamo-target-picker-row` per dimension (a `dynamo-target-picker-dim`
label plus radio `<input>` + `<label>` pairs), and one `dynamo-target-picker-summary` per combination
tagged with `data-sku` / `data-usecase` / `data-variant`; tag every variant-scoped section and
Expected-Performance `<tr>` with the same `data-*`. A **single-target** page uses the static form — no
radios, no `data-*`:

```jsx
<div className="dynamo-target-picker static">
<p className="dynamo-target-picker-title">Deployment target</p>
<div className="dynamo-target-picker-summary">
<span><b>Checkpoint</b> Qwen/Qwen3-8B · BF16</span>
<span><b>Hardware</b> 2x H100 · vLLM · aggregated</span>
</div>
</div>
```

#### Move, defer, or remove a recipe

A catalog page is a triple (page + entry + nav) — never touch just one part:

- **Rename or move**: rename `_catalog/<id>.yaml` and its `id:` together, update the `page:` path, the
  `<id>` in `index.yaml`, the `- page:` in `docs/index.yml`, and the landing card; add a
  `fern/docs.yml` redirect for the old URL.
- **Defer** (hold off the rendered surface): drop `page:` from the entry, add `deferred_reason`, move
  the `<id>` from `recipes:` to `deferred_recipes:` in `index.yaml`, and delete the `.mdx` page, its
  nav `- page:`, and its landing card.
- **Remove**: delete the `.mdx`, the `_catalog/<id>.yaml`, the `index.yaml` entry, the nav `- page:`,
  and the landing card; update the model/target counts; add a redirect.

Run `python3 docs/recipes/_catalog/validate.py` after any of these.

### Add an Example or Recipe (code)

These live **outside `docs/`**, so their READMEs use the HTML-comment SPDX form (no frontmatter), and
docs link to them with absolute GitHub URLs.

- **Example** (`examples/<topic>/`): code-first directory with a `README.md`. Surface it from the
  relevant `docs/<area>/*-examples.md` page.
- **Recipe** (`recipes/<model>/`): `README.md` + `model-cache/` + `<framework>/<mode>/deploy.yaml`
  (+ optional `perf.yaml`). Add a row to the right table in
  [`recipes/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md) — **Feature
  Comparison**, **Aggregated & Disaggregated**, **Functional (Not Yet Benchmarked)**, or
  **Experimental** — per
  [`recipes/CONTRIBUTING.md`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/CONTRIBUTING.md).
  A customer-visible *rendered* recipe page is the separate catalog operation above.

---

## Callouts

Write admonitions GitHub-style; the Fern build auto-converts them (don't hand-write `<Note>`). Put
images under `docs/assets/img/` with descriptive alt text.

| GitHub Syntax | Fern Component |
|---|---|
| `> [!NOTE]` | `<Note>` |
| `> [!TIP]` | `<Tip>` |
| `> [!IMPORTANT]` | `<Info>` |
| `> [!WARNING]` | `<Warning>` |
| `> [!CAUTION]` | `<Error>` |

## Navigation: Tabs and Sections

**`docs/index.yml` is the source of truth — read it for the live structure.** The section names below
are a snapshot, not an authority; sections get added, renamed, and removed. What stays stable is the
*grammar*:

- Two tabs under `navigation:`. **`- tab: docs`** holds the main documentation; **`- tab: recipes`**
  is a flat list of `- page:` entries (`recipes/<slug>.mdx`), order mirroring
  `docs/recipes/_catalog/index.yaml`.
- In the docs tab, each section is marked by a banner comment
  (`# ==================== <Section> ====================`); a `- page:` sits under that section's
  `contents:` at 2-space indent, `path:` relative to `docs/`. In the recipes tab a `- page:` sits
  directly under `layout:`.
- Pages can carry `slug:` (overrides the label-derived slug) and `hidden: true` (reachable by URL but
  off the sidebar — used for per-benchmark pages).

Docs-tab sections **as of this writing** (confirm against `index.yml`): Getting Started, Resources,
Feature Benchmarks, Digest, Kubernetes Deployment, Feature Guides, Backends, Components, Integrations,
Design Docs, Documentation, Hidden Pages. To place a page, match the nearest existing page (see
[Add a Page](#add-a-page)) rather than reasoning from these names.

## Translations and Versioned Navs

- **Chinese translations** live at `fern/translations/zh-CN/pages-dev/<path>`, mirroring the
  English page at `docs/<path>` (same file name and SPDX header, Chinese frontmatter, no body H1 —
  the frontmatter `title` renders the heading, no manual language-switcher links). Fern's native
  localization pairs them and adds the header language picker; untranslated pages fall back to
  English. Links to translated siblings stay shallow-relative; links to untranslated pages are
  deep-relative into `docs/` — count `../` as 4 plus one per directory level of the page under
  `pages-dev/` (`getting-started/x.md` → 5, `components/router/x.md` → 6) — so the repo link
  checker and GitHub browsing stay valid; the sync workflow rewrites them to site URLs at publish
  via `fern/resolve_translation_links.py`. Image refs stay shallow-relative (`../assets/...`) and
  are **not** copied into the mirror — Fern resolves them against the base page. Translate prose, not code, flags, or terminology
  (vLLM / SGLang / TensorRT-LLM stay verbatim). Keep it in sync when the English page changes,
  or don't ship it stale.
- **Versioned navs.** Author only against `docs/` on `main` (the `pages-dev` set). When a release is
  cut, the publish step builds `pages-vX.Y.Z/` from the tagged `docs/` tree and rewrites nav paths —
  **never** edit a `pages-vX.Y.Z/` directory by hand. Write portable paths so the rewrite stays clean.
### Redirects and the version model

The site serves the same nav under three prefixes: **`dev`** (slug `dev`, tracks `main`, regenerated on
every push), **Latest** (slug `/` — the unversioned root `/dynamo/...` *and* `/dynamo/latest/...`, a
frozen snapshot of the newest release), and pinned **`vX.Y.Z`** (immutable snapshots). A
`docs/index.yml` edit on `main` regenerates **only the `dev` nav**.

So a moved or renamed page (changed section or `page:` label) changes only its `/dynamo/dev/<old>` URL.
Add one dev-scoped `fern/docs.yml` redirect:

```yaml
- source: "/dynamo/dev/<old>"
  destination: "/dynamo/dev/<new>"
```

**Do not** add unversioned (`/dynamo/<old>`) or `/dynamo/latest/<old>` redirects for a main-only move:
Latest is frozen, still serves the old path, and a redirect there would break a working URL and point at
a `<new>` that won't exist in Latest until the next release re-snapshots it. Per-version redirects are a
release-time concern, not an authoring one.

## Validate

Self-check, then run the tooling.

**Before you commit**, confirm every must-fix rule in
[Style Guide Is the Standard](#style-guide-is-the-standard) holds for each file you touched — SPDX
header, frontmatter key + no body `# H1`, a nav entry under the right tab, the link rules, and no
internal or sensitive references — and that every internal link and `#anchor` resolves. The docs bot
fails the PR on any of these.

**Tooling:**

```bash
fern check                          # nav + frontmatter structure
fern docs broken-links              # link resolution
python3 docs/recipes/_catalog/validate.py   # recipe/benchmark changes only — validates BOTH catalogs
```

`fern check` and `broken-links` mirror the PR checks. The catalog validator is **not yet wired into
CI**, so run it by hand for any `_catalog/` change. Optional local preview: `fern docs dev`
(localhost:3000, hot reload, no token).

## Commit

```bash
git add docs/ fern/docs.yml          # also recipes/ examples/ fern/main.css when touched
git commit -s -m "docs: <add|update|move|remove> <page-title>"
```

## Debugging

| Symptom | Fix |
|---|---|
| Duplicate H1 on the page | Remove the body `# H1`; Fern renders the title from the nav `page:` |
| SPDX line shows as a heading | Move SPDX inside the `---` frontmatter; add a real metadata key |
| `fern check` YAML error | Check 2-space indent; `- page:` must sit under a section's `contents:` |
| Missing/orphaned file | `path:` in `index.yml` must match the actual file location |
| Broken links in CI | `grep -rn "<filename>" docs/` and fix stale references |
| 404 after a move/rename | Add a **dev-scoped** `fern/docs.yml` redirect (`/dynamo/dev/<old>` → `/dynamo/dev/<new>`); don't redirect `latest`/unversioned (those serve the frozen newest release) |
| MDX parse error | Replace `<https://...>` with `[text](https://...)`; escape stray `<`/`>`; blank line after `<div ...>` and before `</div>`, code fences at column 0 |
| Page missing from site | Ensure the nav entry exists in `index.yml`; allow a few minutes for sync |
| Target picker renders but filters nothing | Use `className` (not `class`) and the exact `dynamo-target-picker` classes; and ensure the axis `value=` is in `fern/main.css` (add its hide rule) |
| `validate.py` fails (orphan/dangling/id) | `_catalog/<id>.yaml` filename, internal `id:`, and the `index.yaml` entry must all match; every deploy/perf asset path must resolve |
| Recipe page absent from the Recipes tab | Add the `- page:` under `- tab: recipes` **and** the `<id>` to `_catalog/index.yaml` |

## Key References

| File | Purpose |
|---|---|
| `docs/documentation-style-guide.md` | Authoring standard for every page (must-fix + guidance) |
| `docs/recipes/_catalog/README.md` | Recipe/benchmark page authoring (catalog contract, blueprint, picker) |
| `docs/recipes/_catalog/validate.py` | Catalog validator (covers both recipe and benchmark catalogs) |
| `docs/index.yml` | Navigation tree (two tabs: `docs` + `recipes`) |
| `docs/` | Content directory (`.md`, plus `.mdx` for recipe/benchmark pages) |
| `docs/assets/` | Images, SVGs, fonts |
| `fern/docs.yml` | Fern site configuration + `redirects:` |
| `fern/main.css` | Pure-CSS target-picker axis values (recipe/benchmark pages) |
| `fern/convert_callouts.py` | Callout conversion (GitHub -> Fern) |
| `recipes/README.md` | Available Recipes tables (code recipes) |
| `recipes/CONTRIBUTING.md` | How to contribute a code recipe |
| `docs/README.md` | Docs system guide (build, sync, publish) |
