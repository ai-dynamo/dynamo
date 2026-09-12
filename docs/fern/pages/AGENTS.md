<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions — docs content and placement

This directory holds every page on the Dynamo docs site. It covers **where a page goes**;
[`../AGENTS.md`](../AGENTS.md) covers **how a page is written** (SPDX, frontmatter, callouts, links),
and the [documentation style guide](community/contributing/documentation/documentation-style-guide.md)
is the full standard. Read all three before adding a page.

The site is tab-based. Placing a page in the wrong tab is the most common and most expensive mistake
here, because the fix is a move plus a redirect.

## Tabs

Nine tabs, each rooted at one directory. The nav key in `docs/fern/index.yml` and the directory name
differ for two of them — match on the directory.

| Tab (nav key) | Directory | Holds |
|---|---|---|
| `home` | `home/` | The landing page. Do not add pages here. |
| `kubernetes-guide` | `kubernetes/` | Deploying and operating Dynamo **on Kubernetes** |
| `cli-guide` | `cli/` | Running Dynamo **from the CLI** on local or bare-metal hosts |
| `use-cases` | `use-cases/` | Workload-shaped guides (agents, multimodal, diffusion, RL, tool calling) |
| `recipes` | `recipes/` | Model recipes, deployment templates, feature benchmarks |
| `developer-guide` | `developer-guide/` | Internals, architecture, customization, contributor-facing knowledge |
| `reference` | `reference/` | Exact contracts: APIs, CRDs, flags, metrics, releases, compatibility |
| `blog` | `blog/` | Dated posts under a year directory |
| `community` | `community/` | Contributing, governance, community process |

## The Kubernetes / CLI split

`kubernetes/` and `cli/` are **parallel guides for two different readers**, not a topic hierarchy.
They share most section names:

```text
kubernetes/  getting-started  installation  model-deployment  kv-aware-routing
             disaggregated-serving  kv-cache-offloading  operations
             fault-tolerance  auto-deployment        <- Kubernetes only

cli/         getting-started  installation  model-deployment  kv-aware-routing
             disaggregated-serving  kv-cache-offloading  operations
```

A feature page belongs to the surface its instructions target:

- Manifests, Helm, CRDs, operator behavior, or `kubectl` → `kubernetes/`.
- `dynamo` / `python3 -m dynamo.*` commands, local processes, env vars → `cli/`.
- Both, in substance → **two pages**, one per tab, each complete for its reader. Do not write one
  page that branches on deployment surface halfway through, and do not cross-link a CLI reader into
  the Kubernetes tab for a step they need.
- Neither — the contract itself, independent of how it is launched → `reference/`.

## Choosing a tab

Work down this list and stop at the first match:

1. Is it an exact contract (flag, field, endpoint, metric, CRD, version matrix)? → `reference/`
2. Is it a dated announcement or narrative post? → `blog/`
3. Is it about contributing to Dynamo itself? → `community/`
4. Is it a model recipe, a deployment template, or a benchmark? → `recipes/`, and follow
   [`recipes/_catalog/README.md`](recipes/_catalog/README.md) — those pages are a triple (page +
   catalog entry + nav), never a lone file.
5. Is it organized around a workload rather than a deployment step? → `use-cases/`
6. Is it internals, architecture, or customization for someone extending Dynamo? → `developer-guide/`
7. Otherwise it is a deployment or operations task → `kubernetes/` or `cli/` per the split above.

Then place the file next to the **nearest existing page on the same topic** within that tab, and
reuse that sibling's subdirectory. Read `docs/fern/index.yml` for the live structure; the section
lists in this file are a snapshot and will drift.

Prefer extending an existing page over adding a new one. A new page has to earn a nav slot.

## Navigation

`docs/fern/index.yml` is the source of truth and the only place a page becomes reachable. A page not
in the nav is dead content.

- The file opens with a `tabs:` map (`display-name`, `icon`, and either `slug:` or `skip-slug: true`),
  then a `navigation:` list of `- tab: <key>` entries, each with a `layout:`.
- Under `layout:`, content is either a `- section:` with `contents:` (sections nest) or a bare
  `- page:`. Sections are marked by a banner comment:
  `# ==================== <Section> ====================`.
- `path:` is relative to `docs/fern/`, so it always starts with `pages/`.
- `- link:` points at a URL instead of a file — used to surface one tab's page from another. It does
  not move the page.
- Page keys worth knowing: `slug:` overrides the label-derived slug, `hidden: true` keeps a page
  reachable by URL but off the sidebar, `collapsed: open-by-default` expands a section, `icon:` sets
  the sidebar glyph.

### URLs and redirects

A page's URL is built by joining the slug of each nav level that contributes one: tab, then section
(sections nest), then the page. Slugs come from the `- page:` / `- section:` **label**, not the file
path, unless an explicit `slug:` overrides it. A level carrying `skip-slug: true` contributes no
segment — but its children still do, so a page under a skip-slug tab lands under its *section* slug,
not at the site root. `pages/developer-guide/advanced-customizations/building-from-source.md` serves
at `/dynamo/dev/advanced-customizations/building-from-source`: the tab is skipped, the section is
not.

Two consequences:

- Renaming a `- page:` label changes the URL even when the file does not move.
- Moving a file between directories changes nothing unless the label, section, or tab changes.

Redirect only when the URL actually changes — a tab, section, label, or explicit `slug:` moved. A
file-only `git mv` that leaves all four alone needs no redirect, and adding one produces a
self-redirect or points at a slug that does not exist. When the URL does change, or when a page is
deleted, add a **dev-scoped** redirect to `redirects:` in `docs/fern/docs.yml` (`/dynamo/dev/<old>`
→ `/dynamo/dev/<new>`). Do not redirect the unversioned or `/dynamo/latest/` forms — those serve a
frozen release snapshot that `main` edits do not touch, and a redirect there breaks a working URL.

Deleting a page also orphans any redirect that already pointed at it. Before removing a page, grep
`docs/fern/docs.yml` for its URL as a `destination:` and retarget every hit.

## Assets and translations

- Images live in `docs/fern/assets/img/`, referenced by a relative path from the page
  (`../../../assets/img/<name>.svg`) with descriptive alt text. Blog posts use their own
  `blog/_assets/` tree instead.
- Chinese translations mirror the English page at
  `docs/fern/translations/zh-CN/pages/<same relative path>` — same file name, same SPDX header,
  Chinese frontmatter, no body `# H1`. Fern's native localization pairs them and falls back to
  English for untranslated pages. Translate prose only: code, flags, and product names
  (vLLM / SGLang / TensorRT-LLM) stay verbatim. A stale translation is worse than a missing one.

## Validate

```bash
python3 docs/fern/scripts/docs_lint.py --scan docs              # SPDX, frontmatter, links, nav coverage
fern check                                            # nav + frontmatter structure
fern docs broken-links                                # link resolution
python3 docs/fern/pages/recipes/_catalog/validate.py  # recipe or benchmark changes only
```

The first three mirror the pre-merge jobs — `Docs Lint`, `Fern Configuration Check`, and
`Fern Broken Links Check`. The catalog validator is not wired into CI — run it by hand for any
`_catalog/` change.
