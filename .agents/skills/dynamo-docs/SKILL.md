---
name: dynamo-docs
description: Adds, updates, moves, or removes content on the Dynamo Fern docs site — standard docs pages, catalog-driven recipe and feature-benchmark pages, examples, recipes, and translations — keeping everything in line with the documentation style guide. Use for any change under docs/, recipes/, or examples/ (new page, edit, tab or section move, rename, removal, recipe/benchmark page, .zh-CN translation, version cut), when deciding which docs tab a page belongs in (Kubernetes Guide vs CLI Guide vs Reference vs Use Cases), and whenever content needs its frontmatter, headings, links, callouts, or terminology fixed.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - docs
    - fern
    - style-guide
---

# Dynamo Docs Maintenance

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Unified skill for adding, updating, moving, and removing content on the Dynamo Fern documentation
site, in line with the project's authoring guides.

Two authoring guides govern this work; read whichever applies before writing:

- [`docs/fern/pages/community/contributing/documentation/documentation-style-guide.md`](../../../docs/fern/pages/community/contributing/documentation/documentation-style-guide.md) — the standard for **every** page: frontmatter, headings, prose, terminology, links, callouts. The must-fix subset is distilled in [Style Guide Is the Standard](#style-guide-is-the-standard) and [Content Rules](#content-rules) below.
- [`docs/fern/pages/recipes/_catalog/README.md`](../../../docs/fern/pages/recipes/_catalog/README.md) — the standard for **recipe and feature-benchmark pages** (the catalog contract, the `.mdx` page blueprint, and the pure-CSS target picker). See [Add a Recipe or Feature Benchmark Page](#add-a-recipe-or-feature-benchmark-page).

## Branch Rule

**ALL edits happen on `main` (or a feature branch based on `main`).**
The `docs-website` branch is CI-managed and must **never** be edited by hand.

## Style Guide Is the Standard

Every page under `docs/` (and the READMEs under `examples/` and `recipes/`) follows the
[Documentation Style Guide](../../../docs/fern/pages/community/contributing/documentation/documentation-style-guide.md)
(`docs/fern/pages/community/contributing/documentation/documentation-style-guide.md`). Read it before
writing content. The `Docs Lint` job (`docs/fern/scripts/docs_lint.py`) enforces a **must-fix** subset on every
PR — get these right or the checks fail:

- **SPDX header** on every file, copyright range `2025-2026`. Fern pages put the two `#` lines
  *inside* the `---` frontmatter; plain READMEs use an HTML-comment block.
- **Frontmatter with at least one metadata key** (`title`/`subtitle`/`sidebar-title`) and **no body
  `# H1`**. Fern renders the page H1 from the nav `page:` value, so a body `# H1` produces a
  duplicate title — and a bare `#` SPDX line left in the body also renders as an H1. Start the body
  at `##`.
- **A nav entry** in `docs/fern/index.yml`, under the right tab, for every new page — a page not in
  the nav is unreachable.
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

- **Page type (Diátaxis).** Each page serves one need — *tutorial* (a tab's `getting-started/`),
  *how-to* (a tab's feature or operations directory), *reference* (`pages/reference/`, for
  flags/APIs/config), or *explanation* (`pages/developer-guide/`). Don't blend a how-to into a flag
  reference; split and cross-link.
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

1. **Pick the tab, then the sibling.** Choose the tab from
   [Navigation](#navigation-tabs-and-sections) — this is the decision that matters, because fixing it
   later costs a move plus a redirect. Then open `docs/fern/index.yml`, find the existing page closest
   in topic to yours *within that tab*, and join **that** section, putting your file in that sibling's
   subdirectory. Page *type* narrows the field (tutorial → the tab's `getting-started/`, how-to → a
   feature or operations directory, reference → `pages/reference/`, explanation →
   `pages/developer-guide/`), but the nearest existing page is the tie-breaker — read the file, don't
   guess from section names. Note the tab, the section, the subdirectory, a kebab-case filename, and
   the page title.
2. Create `docs/fern/pages/<tab-dir>/<subdirectory>/<filename>.md` (use `.mdx` if the page needs Fern
   components). Frontmatter carries the SPDX header plus at least one metadata key; the body starts
   at `##` with a short intro — **no body `# H1`**:

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

3. Add a nav entry in `docs/fern/index.yml` under the section you chose in step 1 — a `- page:` in that
   section's `contents:`, 2-space indent, `path:` relative to `docs/fern/` so it always starts with
   `pages/` (see [Navigation](#navigation-tabs-and-sections) for the grammar):

```yaml
- page: <Page Title>
  path: pages/<tab-dir>/<subdirectory>/<filename>.md
```

### Update a Page

1. Locate by file path, page title, or keyword search (`grep -rn` in `docs/fern/pages/`).
2. **Content only** -- edit the markdown file directly; keep it within the style guide.
3. **Title/label change** -- update the frontmatter (`title`/`sidebar-title`) and the `- page:` name
   in `docs/fern/index.yml`.
4. **Section or tab move** -- `git mv` the file when the directory changes, move the nav entry to the
   new section (and tab), and update every incoming link.

> [!IMPORTANT]
> A page's URL joins the slug of every nav level that contributes one — tab, then section (sections
> nest), then the page. Each slug comes from the nav **label**, not the file path, unless an explicit
> `slug:` overrides it or that level carries `skip-slug: true`. A skipped tab still leaves its
> sections in the URL: `pages/developer-guide/advanced-customizations/building-from-source.md` serves
> at `/dynamo/dev/advanced-customizations/building-from-source`. So **renaming a label changes the URL
> even when the file doesn't move**, and moving a file between directories changes nothing unless its
> label, section, or tab changes. Add a redirect **only when the URL actually changes** — a file-only
> `git mv` that leaves the tab, section, label, and explicit `slug:` alone needs none, and adding one
> yields a self-redirect or a slug that doesn't exist. When the URL does change, add a
> **dev-scoped** redirect to the `redirects:` list in `docs/fern/docs.yml`: `/dynamo/dev/<old>` →
> `/dynamo/dev/<new>`. Editing `docs/fern/index.yml` regenerates only the `dev` nav, so do **not** redirect
> the unversioned (`/dynamo/<old>`) or `/dynamo/latest/<old>` forms — those serve **Latest**, a frozen
> release snapshot that `main` edits don't touch, and a redirect there would break a working URL. See
> [Redirects and the version model](#redirects-and-the-version-model).

### Remove a Page

Removing a page is destructive and breaks live URLs. Confirm with the user before step 2, and show
them the incoming links and redirects you found in steps 1 and 2.

1. Find incoming links: `grep -rn "<filename>" docs/`.
2. Find redirects that already point at the page: grep its published URL in `docs/fern/docs.yml` as
   a `destination:`. Each hit has to be retargeted, or it starts serving a 404.
3. Remove the file, matching its real extension — pages are `.md` or `.mdx`:
   `git rm docs/fern/pages/<tab-dir>/<subdirectory>/<filename>.<ext>`.
4. Remove the `- page:` block from `docs/fern/index.yml`. If it was the last page in a section, remove the
   whole `- section:` block.
5. Fix or remove every incoming link found in step 1, retarget every redirect found in step 2, and add
   a `docs/fern/docs.yml` redirect for the page's own URL if it had a stable one.

### Add a Recipe or Feature Benchmark Page

Recipe and feature-benchmark pages are **catalog-driven** and use `.mdx` (they embed a pure-CSS
target picker). Authoritative guide:
[`docs/fern/pages/recipes/_catalog/README.md`](../../../docs/fern/pages/recipes/_catalog/README.md).
Each page is a triple — page + catalog entry + nav:

1. **Write the `.mdx`** at `docs/fern/pages/recipes/model-recipes/<slug>.mdx` (or `docs/fern/pages/recipes/feature-benchmarks/<slug>.mdx`). Frontmatter
   carries SPDX + `title` + one-sentence `subtitle`; body starts with a short intro, then the target
   picker — multi-target pages use the radio picker, single-target pages use the **static** form
   (exact classes under [Target picker](#target-picker) below) — then the fixed section order:
   `## Prerequisites` → `## Deploy` → `## Smoke Test` → `## Benchmark` → `## Expected Performance`
   (omit if no numbers) → `## Compare All Targets` (multi-target only) → `## Related Feature
   Benchmarks` → `## Notes` → `## Source`. **MDX rule:** blank line after `<div ...>` and before
   `</div>`; keep code fences at column 0.
2. **Add a catalog entry** — one file at `docs/fern/pages/recipes/_catalog/recipes/<id>.yaml` (or
   `docs/fern/pages/recipes/feature-benchmarks/_catalog/benchmarks/<id>.yaml`), SPDX header, exactly one object. **Read the
   sibling `schema.json` first for the exact field set** (`docs/fern/pages/recipes/_catalog/schema.json` for
   recipes, `docs/fern/pages/recipes/feature-benchmarks/_catalog/schema.json` for benchmarks — they are **different** schemas) —
   each is `additionalProperties: false`, so an invented or misspelled key fails validation; don't
   guess the shape. A **recipe** entry requires `id`,
   `title`, `provider`, `model`, `status`, `targets`, `maintainer`, and each `targets[]` item
   requires `id`, `recommended`, `hardware`, `runtime`, `topology`, `techniques`, `workload`,
   `deploy`, `expected_performance`. Internal `id:` **must equal the filename**; active entries carry
   `page:`, deferred ones carry `deferred_reason` and omit `page:`. Add the `<id>` to the matching
   `_catalog/index.yaml` (`recipes:` for active, `deferred_recipes:` for deferred — it controls
   sidebar/landing order).
3. **Wire navigation** in `docs/fern/index.yml`: everything here lives under `- tab: recipes` — a
   `- page:` in the **Model Recipes** section for recipes, or in the **Feature Benchmarks** section
   for benchmarks. Per-benchmark pages are usually `hidden: true` (surfaced from the landing page).
4. **Patch `docs/fern/main.css` only if** the page introduces a picker axis value not already supported
   (`recipe-sku`: `b200`/`h200`/`h100`/`gb200`/`hopper`/`blackwell`; `recipe-usecase`:
   `chat`/`agentic`; `recipe-variant`: `agg`/`disagg`/…). A value missing from CSS renders but
   filters nothing. After editing `main.css`, run `python3 docs/fern/scripts/sync_site_css.py` so the
   footer's CSS mirror stays in sync — pre-commit fails otherwise.
5. **Add the landing card** in `docs/fern/pages/recipes/model-recipes/overview.mdx` and update the model/target counts.
6. **Validate**: `python3 docs/fern/pages/recipes/_catalog/validate.py` (covers both catalogs), then `fern
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

**Benchmarks use a different schema.** A `docs/fern/pages/recipes/feature-benchmarks/_catalog/benchmarks/<id>.yaml` entry
validates against `docs/fern/pages/recipes/feature-benchmarks/_catalog/schema.json`, whose required set is `id`, `title`, `page`,
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
  `<id>` in `index.yaml`, the `- page:` in `docs/fern/index.yml`, and the landing card; add a
  `docs/fern/docs.yml` redirect for the old URL.
- **Defer** (hold off the rendered surface): drop `page:` from the entry, add `deferred_reason`, move
  the `<id>` from `recipes:` to `deferred_recipes:` in `index.yaml`, and delete the `.mdx` page, its
  nav `- page:`, and its landing card.
- **Remove**: delete the `.mdx`, the `_catalog/<id>.yaml`, the `index.yaml` entry, the nav `- page:`,
  and the landing card; update the model/target counts; add a redirect.

Run `python3 docs/fern/pages/recipes/_catalog/validate.py` after any of these.

### Add an Example or Recipe (code)

These live **outside `docs/`**, so their READMEs use the HTML-comment SPDX form (no frontmatter), and
docs link to them with absolute GitHub URLs.

- **Example** (`examples/<topic>/`): code-first directory with a `README.md`. Surface it from the
  relevant `*-examples.md` page (component-scoped ones live under `pages/developer-guide/`) or from
  the topic page that needs it. There is no general Examples landing page — the empty
  `pages/reference/general/examples.md` stub was removed, and `/dynamo/dev/reference/examples` now
  redirects to the recipes catalog. Don't recreate it.
- **Recipe** (`recipes/<model>/`): `README.md` + `model-cache/` + `<framework>/<mode>/deploy.yaml`
  (+ optional `perf.yaml`). Add a row to the right table in
  [`recipes/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md) — **Feature
  Comparison**, **Aggregated & Disaggregated**, **Functional (Not Yet Benchmarked)**, or
  **Experimental** — per
  [`recipes/CONTRIBUTING.md`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/CONTRIBUTING.md).
  A customer-visible *rendered* recipe page is the separate catalog operation above.

---

## Callouts

Match admonition syntax to the extension: use Fern callout components in `.mdx`, and GitHub-style blockquotes in `.md`. Put
images under `docs/fern/assets/img/` with descriptive alt text, referenced by a relative path from the
page (`../../../assets/img/<name>.svg`). Blog posts use their own `pages/blog/_assets/` tree instead.

| GitHub Syntax | Fern Component |
|---|---|
| `> [!NOTE]` | `<Note>` |
| `> [!TIP]` | `<Tip>` |
| `> [!IMPORTANT]` | `<Info>` |
| `> [!WARNING]` | `<Warning>` |
| `> [!CAUTION]` | `<Error>` |

## Navigation: Tabs and Sections

**`docs/fern/index.yml` is the source of truth — read it for the live structure.** The section names below
are a snapshot, not an authority; sections get added, renamed, and removed. What stays stable is the
*grammar*:

- The file opens with a `tabs:` map — each tab key carries `display-name`, `icon`, and either a
  `slug:` or `skip-slug: true` — then a `navigation:` list of `- tab: <key>` entries, each with a
  `layout:`.
- Under `layout:`, content is either a `- section:` with `contents:` (sections nest) or a bare
  `- page:`. Sections are marked by a banner comment
  (`# ==================== <Section> ====================`).
- `path:` is relative to `docs/fern/`, so it always starts with `pages/`. 2-space indent.
- `- link:` points at a URL rather than a file — used to surface one tab's page from another tab's
  sidebar. It does not move the page.
- Pages can carry `slug:` (overrides the label-derived slug) and `hidden: true` (reachable by URL but
  off the sidebar — used for per-benchmark pages); sections can carry `collapsed: open-by-default`.

Nine tabs, each rooted at one directory under `docs/fern/pages/`. The nav key and the directory name
differ for the two guide tabs — match on the directory:

| Tab (nav key) | Directory | Holds |
|---|---|---|
| `home` | `pages/home/` | The landing page. Don't add pages here |
| `kubernetes-guide` | `pages/kubernetes/` | Deploying and operating Dynamo **on Kubernetes** |
| `cli-guide` | `pages/cli/` | Running Dynamo **from the CLI** on local or bare-metal hosts |
| `use-cases` | `pages/use-cases/` | Workload-shaped guides (agents, multimodal, diffusion, RL, tool calling) |
| `recipes` | `pages/recipes/` | Model recipes, deployment templates, feature benchmarks |
| `developer-guide` | `pages/developer-guide/` | Internals, architecture, customization |
| `reference` | `pages/reference/` | Exact contracts: APIs, CRDs, flags, metrics, releases, compatibility |
| `blog` | `pages/blog/` | Dated posts under a year directory |
| `community` | `pages/community/` | Contributing, governance, community process |

**`kubernetes/` and `cli/` are parallel guides for two different readers, not a topic hierarchy.**
They share most section names (Getting Started, Installation, Model Deployment, KV-Aware Routing,
Disaggregated Serving, KV Cache Offloading, Operations; Kubernetes adds Fault Tolerance and Auto
Deployment). Place a feature page by the surface its instructions target: manifests, Helm, CRDs,
operator behavior, or `kubectl` → `kubernetes/`; `dynamo` / `python3 -m dynamo.*` commands, local
processes, env vars → `cli/`. If it genuinely covers both, write **two pages**, one per tab, each
complete for its reader — never one page that branches on deployment surface halfway through. If it's
the contract itself, independent of how it's launched, it belongs in `reference/`.

Within the chosen tab, match the nearest existing page (see [Add a Page](#add-a-page)) rather than
reasoning from section names. Fuller placement guidance lives in
[`docs/fern/pages/AGENTS.md`](../../../docs/fern/pages/AGENTS.md).

## Translations and Versioned Navs

- **Chinese translations** live at `docs/fern/translations/zh-CN/pages/<path>`, mirroring the
  English page at `docs/fern/pages/<path>` (same file name and SPDX header, Chinese frontmatter, no
  body H1 — the frontmatter `title` renders the heading, no manual language-switcher links). Fern's
  native localization pairs them and adds the header language picker; untranslated pages fall back to
  English. Links to translated siblings stay relative within the locale mirror; links to untranslated
  pages point back into the base tree — count `../` as **3** (`pages` → `zh-CN` → `translations`,
  landing at `docs/fern/`) plus one per directory level of the page under `pages/`, then append
  `pages/<path>`. So `cli/getting-started/quickstart.mdx` uses five:
  `../../../../../pages/reference/general/release-artifacts.mdx`. That keeps the repo link checker
  and GitHub browsing valid; the sync workflow rewrites them to site URLs at publish via
  `docs/fern/scripts/resolve_translation_links.py`. Image refs are **not** copied into the mirror —
  Fern resolves them against the base page. Translate prose, not code, flags, or terminology
  (vLLM / SGLang / TensorRT-LLM stay verbatim). Keep it in sync when the English page changes,
  or don't ship it stale.
- **Versioned navs.** Author only against `docs/fern/pages/` on `main`. The sync workflow copies that
  tree to `fern/pages-dev/` on the CI-managed `docs-website` branch; when a release is cut, the
  publish step builds `fern/pages-vX.Y.Z/` from the tagged tree and rewrites nav paths — **never**
  edit `docs-website` or a `pages-vX.Y.Z/` directory by hand. Write portable paths so the rewrite
  stays clean. Translation mirrors snapshot the same way
  (`fern/translations/<lang>/pages-vX.Y.Z/` from the tag's `pages-dev` mirror, links resolved under
  the tag's version slug); tags cut from branches without translations skip the snapshot. To validate
  a change to that composition before merge, replay both jobs with
  `docs/fern/scripts/simulate_docs_website.sh`.

### Redirects and the version model

The site serves the same nav under three prefixes: **`dev`** (slug `dev`, tracks `main`, regenerated on
every push), **Latest** (slug `/` — the unversioned root `/dynamo/...` *and* `/dynamo/latest/...`, a
frozen snapshot of the newest release), and pinned **`vX.Y.Z`** (immutable snapshots). A
`docs/fern/index.yml` edit on `main` regenerates **only the `dev` nav**.

So a moved or renamed page (changed section or `page:` label) changes only its `/dynamo/dev/<old>` URL.
Add one dev-scoped `docs/fern/docs.yml` redirect:

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
internal or sensitive references — and that every internal link and `#anchor` resolves. The
`Docs Lint` job fails the PR on any of these.

**Tooling:**

```bash
python3 docs/fern/scripts/docs_lint.py --scan docs   # SPDX, frontmatter, links, nav, internal refs
fern check                          # nav + frontmatter structure
fern docs broken-links              # link resolution
python3 docs/fern/pages/recipes/_catalog/validate.py   # recipe/benchmark changes only — validates BOTH catalogs
```

`docs_lint.py`, `fern check`, and `broken-links` are the three PR jobs; the first reports through
inline annotations on the offending line. The catalog validator is **not yet wired into CI**, so run
it by hand for any `_catalog/` change. Optional local preview: `fern docs dev` (localhost:3000, hot
reload, no token).

## Commit

```bash
git add docs/fern/pages docs/fern/index.yml docs/fern/docs.yml   # also recipes/ examples/ docs/fern/main.css when touched
git commit -s -m "docs: <add|update|move|remove> <page-title>"
```

## Debugging

| Symptom | Fix |
|---|---|
| Duplicate H1 on the page | Remove the body `# H1`; Fern renders the title from the nav `page:` |
| SPDX line shows as a heading | Move SPDX inside the `---` frontmatter; add a real metadata key |
| `fern check` YAML error | Check 2-space indent; `- page:` must sit under a section's `contents:` |
| Missing/orphaned file | `path:` in `index.yml` must match the actual file location, and is relative to `docs/fern/` (starts with `pages/`) |
| Broken links in CI | `grep -rn "<filename>" docs/` and fix stale references |
| Page landed in the wrong guide | Kubernetes-surface instructions belong in `pages/kubernetes/`, CLI-surface in `pages/cli/`; `git mv` and add a dev-scoped redirect |
| `sync_site_css.py` pre-commit failure | Ran after a `main.css` edit — `python3 docs/fern/scripts/sync_site_css.py` and stage the result |
| 404 after a move/rename | Add a **dev-scoped** `docs/fern/docs.yml` redirect (`/dynamo/dev/<old>` → `/dynamo/dev/<new>`); don't redirect `latest`/unversioned (those serve the frozen newest release) |
| MDX parse error | Replace `<https://...>` with `[text](https://...)`; escape stray `<`/`>`; blank line after `<div ...>` and before `</div>`, code fences at column 0 |
| Page missing from site | Ensure the nav entry exists in `index.yml`; allow a few minutes for sync |
| Target picker renders but filters nothing | Use `className` (not `class`) and the exact `dynamo-target-picker` classes; and ensure the axis `value=` is in `docs/fern/main.css` (add its hide rule) |
| `validate.py` fails (orphan/dangling/id) | `_catalog/<id>.yaml` filename, internal `id:`, and the `index.yaml` entry must all match; every deploy/perf asset path must resolve |
| Recipe page absent from the Recipes tab | Add the `- page:` under `- tab: recipes` **and** the `<id>` to `_catalog/index.yaml` |

## Key References

| File | Purpose |
|---|---|
| `docs/fern/pages/community/contributing/documentation/documentation-style-guide.md` | Authoring standard for every page (must-fix + guidance) |
| `docs/fern/pages/community/contributing/documentation/building-and-publishing.md` | Docs system guide (branch model, sync, publish, versions) |
| `docs/fern/AGENTS.md` | Docs mechanics + the `docs/fern/` file map |
| `docs/fern/pages/AGENTS.md` | Tab taxonomy and page placement |
| `docs/fern/pages/recipes/_catalog/README.md` | Recipe/benchmark page authoring (catalog contract, blueprint, picker) |
| `docs/fern/pages/recipes/_catalog/validate.py` | Catalog validator (covers both recipe and benchmark catalogs) |
| `docs/fern/index.yml` | Navigation tree (nine tabs; `path:` relative to `docs/fern/`) |
| `docs/fern/pages/` | Content directory (`.md` and `.mdx`), one subdirectory per tab |
| `docs/fern/assets/` | Images, SVGs, fonts |
| `docs/fern/translations/` | Locale mirrors of `pages/` (`zh-CN/pages/<same path>`) |
| `docs/fern/scripts/docs_lint.py` | Structural linter behind the `Docs Lint` PR job |
| `docs/fern/docs.yml` | Fern site configuration + `redirects:` |
| `docs/fern/main.css` | Pure-CSS target-picker axis values (recipe/benchmark pages) |
| `docs/fern/scripts/convert_callouts.py` | Callout conversion (GitHub -> Fern) |
| `docs/fern/scripts/sync_site_css.py` | Footer CSS mirror; run after any `main.css` edit |
| `docs/fern/scripts/simulate_docs_website.sh` | Local replay of the sync + release composition |
| `recipes/README.md` | Available Recipes tables (code recipes) |
| `recipes/CONTRIBUTING.md` | How to contribute a code recipe |
