---
name: fern-navigation
description: Knowledge of Fern's site-level navigation and structure configuration — how a docs site is organized in `docs.yml` (and product/version `.yml` files) using sections, pages, folders, tabs, tab variants, versions, products, changelogs, and site-level settings, plus per-page frontmatter. Use when designing or changing the shape of a Fern docs site (adding a tab, splitting into products, cutting a version, restructuring the sidebar, moving the changelog, tuning the navbar/layout/theme) or when a user asks "what options does Fern give me for navigation" or "how do I configure X in docs.yml". Complements fern-components (in-page MDX components) and dynamo-docs (this repo's page placement, style guide, and .md/.mdx rules).
---

# Fern Navigation & Site Structure

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Fern documentation is configured at two levels: **site-level** in `docs.yml` (and, for multi-product
or multi-version sites, in per-product / per-version `.yml` files), and **page-level** in each `.md` /
`.mdx` file's frontmatter. This skill catalogs the whole site-structure surface — what each building
block does, **when to reach for it**, and the exact YAML — and points to
[`references/navigation-reference.md`](references/navigation-reference.md) for every field, default,
and copy-paste example.

This skill is **structure & configuration knowledge**. In-page MDX components (`<Card>`, `<Steps>`,
`<Tabs>` content tabs, `<Accordion>`, callouts…) belong to the **fern-components** skill. Page
placement conventions, the style guide, SPDX headers, terminology, and this repo's `.md`-vs-`.mdx`
rule belong to the **dynamo-docs** skill. Use all three together when authoring.

## "tabs" and "tabs" are two different things (read this first)

Fern overloads several words. Keep the layers straight before touching config:

| Term | Layer | Where it lives | Owned by |
|---|---|---|---|
| **`<Tabs>` / `<Tab>`** | in-page content tabs | inside a `.mdx` body | fern-components |
| **`tabs:` navigation** | site-level nav tabs | `docs.yml` top-level + `navigation:` | **this skill** |
| **`<Versions>` component** | in-page version-conditional content | inside a `.mdx` body | fern-components |
| **`versions:` config** | site-wide version switcher | `docs.yml` + `versions/*.yml` | **this skill** |

## The mental model: how a URL is built

Fern composes each page's URL by concatenating a slug from **every level** of the hierarchy it passes
through, outermost first:

```
/<product>/<version>/<tab>/<section>/<folder>/<page>
```

Each level auto-generates a slug from its display name (or filename, for folders). Any level can
`slug:` rename itself, `skip-slug: true` drop itself from the path, or a page's frontmatter `slug:`
override the whole section/folder portion (product/version prefix is preserved). Understanding this
composition is the key to predicting and controlling URLs — see the reference's Slugs section.

## Choosing the right structural tool

Decide by **how different two bodies of content are**, from lightest to heaviest:

| You want to… | Reach for | Notes |
|---|---|---|
| Order pages in the sidebar | **sections / pages / folders** | The default. Folders auto-discover files. |
| Auto-build nav from a directory tree | **`folder:`** | `index.md(x)` becomes the section overview; subdirs nest. |
| Group whole content areas (Guides vs API Ref) | **tabs** | Top-level `tabs:`, referenced in `navigation:`. |
| Same area, different audience/perspective (REST vs GraphQL, dev vs PM) | **tab variants** | `variants:` instead of `layout:`; supports RBAC. |
| Multiple releases of the same docs | **versions** | Dropdown switcher; `versions/*.yml`. Team plan. |
| Multiple distinct products under one site | **products** | Product switcher; `products/*.yml`. Team plan. Can nest versions. |
| A dated log of changes | **changelog** | A `changelog/` folder, surfaced as a tab or a section. |
| A root page before any product | **`landing-page:`** | Independent of products/versions. |
| Link out (GitHub, dashboard) from nav | **`href` tab / external product / navbar link** | No internal content. |

Rules of thumb:
- **Don't reach for products when a tab will do.** Products are a heavyweight, Team/Enterprise
  feature that removes top-level `navigation`/`tabs` and splits config into files. Use them only for
  genuinely separate products, not for two sections of one product.
- **Variants vs tabs:** variants = different *lenses on the same area*; tabs = *different areas*.
- **Versions and products compose** — a product can be versioned, and versioned/unversioned products
  coexist. Standalone versioning (no products) is simpler; prefer it if you have one product.

## Hard constraints that bite (verify against these)

- **Products/versions eject top-level nav.** When you add `products:` (or `versions:`), you MUST
  remove the top-level `navigation:` and `tabs:` from `docs.yml` — they move into the per-product /
  per-version files. `fern check` rejects a `navigation` block coexisting with `products:` and would
  otherwise render an empty site.
- **External products / `href` tabs** cannot have `navigation`, `tabs`, `layout`, or `variants`.
- **A tab needs exactly one of** `layout`, `variants`, or `href` — never `href` + content.
- **The default version (first in the list) can't be `hidden`.**
- **`changelog/` must be named exactly that**, files live flat in its root (no subdirs), and file
  names must be dated (`YYYY-MM-DD`, `MM-DD-YYYY`, or `MM-DD-YY`). Section-level changelogs can't nest
  under an `api` entry.
- **Team plan gate:** versions and products are Team/Enterprise features.

## The eight source pages (what's in the reference)

[`references/navigation-reference.md`](references/navigation-reference.md) mirrors the eight pages of
Fern's `navigation/` docs section, with full YAML and every field:

1. **Overview** — the two config levels; index of the structural cards.
2. **Sections, pages, folders** — the `navigation:` tree, nesting, folder auto-discovery, slugs,
   hiding, availability badges, collapsed state, sidebar icons, external links.
3. **Tabs & tab variants** — `tabs:` + `navigation:`, all tab props, variants, placement/style.
4. **Versions** — `versions/*.yml`, the switcher, availability, slugs, audiences, hiding, styling.
5. **Products** — `products/*.yml`, internal vs external, the switcher, versioned products, landing
   page, audiences, conditional content, search scoping, selector CSS.
6. **Changelogs** — `changelog/` folder, tab vs section placement, entry files, tags, layouts, RSS.
7. **Frontmatter (page-level settings)** — every per-page field: titles, slug, description, layout,
   TOC/nav/feedback toggles, availability, SEO/OpenGraph, changelog tags.
8. **Site-level settings** — the full `docs.yml` surface: colors, logo, typography, layout, theme,
   navbar/footer links, instances, settings (search etc.), page actions, redirects, metadata,
   analytics, edit-this-page, Ask Fern, agents/llms.txt, check rules.

## Workflow for a structure change in this repo

1. **Locate the config.** Find this repo's `docs.yml` (and any `products/` or `versions/` files) —
   `dynamo-docs` owns where pages live; this skill owns their arrangement.
2. **Pick the lightest tool** from the table above that expresses the change.
3. **Write the YAML** from the reference, honoring the hard constraints.
4. **Mind the URL impact.** Any slug/section/tab/version/product change alters URLs — add
   `redirects:` for moved pages, and remember `check.rules.missing-redirects`.
5. **Validate:** run `fern check` (broken links, missing redirects, config coherence) before publish.

## Keeping this skill current

Fern's config schema drifts. The source of truth is `fern-api/docs` (branch `main`),
`fern/products/docs/pages/navigation/*.mdx`. `manifest.json` records each tracked page's git blob
SHA as of the last sync. To check for drift and refresh:

```bash
python3 scripts/refresh_navigation.py --check                 # what changed upstream?
python3 scripts/refresh_navigation.py --fetch --out /tmp/fern # download changed pages
# ...update references/navigation-reference.md + this SKILL.md from those...
python3 scripts/refresh_navigation.py --sync                  # record new SHAs as current
```

For anything load-bearing, verify field names against the live
[`docs-yml` schema](https://schema.buildwithfern.dev/docs-yml.json) or the current docs rather than
trusting a summary — this surface changes.
