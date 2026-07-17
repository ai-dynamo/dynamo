---
name: fern-components
description: Knowledge of Fern's built-in MDX component library (accordions, callouts, cards, steps, tabs, code blocks, API-reference snippets, and more) for authoring docs pages. Use when writing or editing a Fern `.mdx` page and deciding whether a component would present content better than plain Markdown, when a user asks "what Fern components exist" or "how do I use `<X>`", or when reviewing a page for missed opportunities to use a component. Complements dynamo-docs (which owns page placement, nav, frontmatter, and the style guide).
---

# Fern Components

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Fern ships a built-in library of ~27 MDX components you can use in documentation pages without
importing anything. This skill catalogs them, says **when each is worth reaching for**, and points to
[`references/components-reference.md`](references/components-reference.md) for exact syntax, every prop,
and copy-paste examples.

This skill is **component knowledge only**. Page placement, `docs/index.yml` nav, frontmatter, SPDX
headers, links, terminology, and the style guide belong to the **dynamo-docs** skill — use both
together when authoring.

## The `.md` vs `.mdx` rule (read this first in this repo)

Fern components are JSX. They render **only in `.mdx` files**. This repo (`ai-dynamo/dynamo`) mixes two
page formats, and the boundary is a hard must-fix:

| Page type | How to write rich content |
|---|---|
| **`.mdx`** (e.g. `getting-started/introduction.mdx`, recipe pages) | Use Fern components **directly** — `<CardGroup>`, `<Steps>`, `<Accordion>`, `<Tabs>`, `<Note>`, etc. |
| **`.md`** (most docs pages) | **Do not hand-write `<Note>`/`<Tip>`/etc.** Write callouts GitHub-style (`> [!NOTE]`); `fern/convert_callouts.py` converts them at build. Other components (Cards, Steps, Tabs…) are **not** available — restructure with plain Markdown, or convert the page to `.mdx` on purpose. |

Callout conversion map (GitHub → Fern), for `.md` pages:

| `> [!NOTE]` | `> [!TIP]` | `> [!IMPORTANT]` | `> [!WARNING]` | `> [!CAUTION]` |
|---|---|---|---|---|
| `<Note>` | `<Tip>` | `<Info>` | `<Warning>` | `<Error>` |

**Before using any component below, confirm the target file is `.mdx`.** If it's `.md` and you need a
non-callout component, the decision (restructure vs. rename to `.mdx`) is a dynamo-docs / nav concern —
raise it, don't silently rename.

## When to reach for a component (and when not)

Prefer plain Markdown. A component earns its place only when it does something Markdown can't:
progressive disclosure, sequencing, branching, live API data, or interactivity. Don't decorate — an
ordered list is better than `<Steps>` for two trivial steps, and a sentence is better than a `<Card>`
whose only job is to host a link (see the dynamo-docs / Fern cross-reference guidance).

| You want to… | Component | Notes |
|---|---|---|
| Flag a note / warning / tip | **Callout** (`<Note>` `<Tip>` `<Warning>` `<Info>` `<Success>` `<Error>` `<Launch>` `<Check>`) | In `.md`, use `> [!NOTE]` syntax instead (see above). |
| Collapse FAQs / optional detail | **Accordion** / **AccordionGroup** | Content stays SEO-indexed while collapsed. |
| Sequence a tutorial / setup | **Steps** / **Step** | Auto-numbered, anchor links. Use `toc` to surface in the TOC. |
| Show the same thing per-language / per-OS | **Tabs** / **Tab** | `language=` syncs all tabs+code blocks site-wide. |
| Navigation grid / feature hub | **Card** / **CardGroup** | `cols={n}`, Font Awesome icons, images, `href` makes the whole card clickable. |
| Rich code (highlight, focus, title, embed a file) | **Code block** / `<Code>` / `<CodeBlocks>` / `<CodeGroup>` | Fenced ``` with attrs; `<Code src>` embeds local/GitHub files. |
| Multiple install commands (npm/pnpm/yarn) | **CodeGroup** with `for=` | Custom sync group independent of language. |
| Image with caption / framing | **Frame** | Wraps `<img>`/`<video>`; `background="subtle"`. |
| Long / searchable / sticky-header table | **StickyTable** / **SearchableTable** / **StickySearchableTable**, or `<table sticky searchable>` | Plain Markdown tables are fine for short data. |
| Inline status / version chip | **Badge** | For longer notes use a Callout instead. |
| Small icon inline / in headings | **Icon** | Font Awesome name or `./path.svg`. |
| Clickable button / CTA / download trigger | **Button** | `intent`, `href`, icons. |
| Downloadable asset (PDF, ZIP bundle) | **Download** | `src=` single file, `sources={[…]}` zips multiple. |
| Click-to-copy inline text | **Copy** | Show one value, copy another via `clipboard=`. |
| Hover explanation for a term or code token | **Tooltip** / `<Template>` | `<Template>` adds tooltips to code-block variables. |
| Document a param / field / config key | **ParamField** | The standard field-doc row: `path`, `type`, `required`, `default`, `deprecated`. |
| Indent nested params visually | **Indent** | Wraps any content (unlike `<Folder>`). |
| Show a project / directory tree | **Files** / **Folder** / **File** | `defaultOpen`, `href`, `highlighted`, `comment`. |
| Link to non-heading content | **Anchor** | `id=` on paragraphs, tables, code blocks. |
| Float supplementary content right | **Aside** | Sticky; good for an endpoint snippet beside prose. |
| Copyable AI prompt (open in Cursor/Claude/ChatGPT) | **Prompt** | `actions={["cursor","claude","chatgpt"]}` or custom URL. |
| Show/hide by product, version, or role | **If** | `products` / `versions` / `roles`, combinable, `not` to invert. |
| Inline versioned content with a switcher | **Versions** / **Version** | Distinct from site-wide versioning. |
| **API reference — request code sample** | **EndpointRequestSnippet** | `endpoint="POST /path"`; `languages`, `payload`, `hideTryItButton`. |
| **API reference — response sample** | **EndpointResponseSnippet** | Pulls from your API definition. |
| **API reference — endpoint schema (params/body)** | **EndpointSchemaSnippet** | `selector="request.body"` etc. |
| **API reference — any named type** | **Schema** / **SchemaSnippet** | `<Schema type="…">` fields; `<SchemaSnippet>` JSON. |
| **API reference — live "try it" request builder** | **RunnableEndpoint** | Real HTTP calls from the page. |
| **API reference — webhook payload** | **WebhookPayloadSnippet** | By `operationId`. |
| Reuse a Markdown fragment in many places | **`<Markdown src>`** (reusable snippets) | Single-source constants/warnings; supports `{{params}}`. |
| Something bespoke / interactive | **Custom React component** | `.tsx` in a components dir wired via `docs.yml`; SSR'd. |

Full syntax, every prop, variants, and examples for **all** of the above are in
[`references/components-reference.md`](references/components-reference.md). Read the relevant section
before writing a component you don't use often — props and exact names (e.g. `<Note>` vs `<Callout
intent>`, `iconSize` math, `selector` values) are easy to get wrong from memory.

## Authoring workflow

1. **Confirm the file is `.mdx`.** If `.md`, only GitHub-style callouts apply (see the rule above).
2. **Ask "does Markdown already do this?"** If yes, use Markdown. Only reach for a component when it
   adds disclosure, sequencing, branching, interactivity, or live API data.
3. **Open the reference** section for the component and copy the closest example. Match prop names and
   casing exactly (MDX is `className`, not `class`; attributes are `camelCase` — `autoPlay`, not
   `autoplay`).
4. **Nest deliberately.** Most containers accept rich children (a `<Note>` inside a `<Step>`, a
   `<CardGroup>` inside an `<Accordion>`). `<Folder>` is the exception — it takes only `<File>` /
   `<Folder>`; use `<Indent>` when you need to indent anything else.
5. **Anchors:** `##`/`###` headings auto-generate anchors; `title=` props on `<Step>`, `<Tab>`,
   `<Accordion>`, `<Card>` **do not**. To deep-link one, add a real heading or use `<Anchor id>`.
6. **MDX gotchas that break the build:** put a blank line after `<div …>` and before `</div>`; keep
   code fences at column 0; replace bare `<https://…>` with `[text](https://…)`; escape stray `<`/`>`
   and literal `$` (`\$`) outside code.

## Suggesting components during review

When reviewing or improving an existing `.mdx` page, watch for these high-value swaps:

- A wall of "first… then… next…" prose or a long ordered list of actions → **Steps**.
- Parallel per-language / per-OS / per-backend blocks repeated back to back → **Tabs** (with
  `language=` when it's code, so they sync).
- A long FAQ, or optional deep-dive detail interrupting the main flow → **Accordion**.
- A cluster of "see also" links at the top or bottom of a page → **CardGroup** of **Card**s.
- Hand-maintained request/response code that duplicates the API definition → the **Endpoint*Snippet**
  family (also gives AI agents structured Markdown output).
- The same constant, warning, or boilerplate copy-pasted across pages → a **reusable snippet**
  (`<Markdown src>`).
- A very long reference table users scan → **StickyTable** / **SearchableTable**.

Suggest the lightest option, name the component, and point at the reference section — don't rewrite a
working page just to add components.

## Staying in sync with Fern (refresh mechanism)

This skill is a **cached snapshot** of Fern's component docs, not a live mirror — so it works offline
and loads instantly. The trade-off is that it can drift when Fern ships or changes a component.
[`scripts/refresh_components.py`](scripts/refresh_components.py) tracks that drift, and
[`manifest.json`](manifest.json) records the upstream git-blob SHA of every source page as of the last
sync (see `manifest.json`'s `fetched_at` for how old this snapshot is).

Source of truth (GitHub `fern-api/docs@main`): the component-library pages under
`fern/products/docs/pages/component-library/**` plus `customization/custom-react-components.mdx` and
`navigation/tabs.mdx`.

**On-load staleness check (do this when the skill loads):** read `manifest.json`'s `fetched_at`. If it
is more than ~3 months old — or the user asks whether the components are current, or a component
doesn't behave as the reference claims — tell the user the snapshot's age and offer to refresh. Don't
refresh unprompted: it needs internet and edits the reference, so surface it and let the user say go.
Refreshing prefers the `gh` CLI and falls back to the public GitHub API; everything else here is
offline.

```bash
# From the skill dir. --check is read-only and safe to run anytime.
python3 scripts/refresh_components.py --check              # drift vs manifest? exit 2 = drift, 0 = in sync
python3 scripts/refresh_components.py --fetch --out /tmp/fern-refresh   # download only changed pages
#   → then update references/components-reference.md and the tables above from those pages,
python3 scripts/refresh_components.py --sync               # record the new SHAs as current
```

Only `--sync` writes the manifest; `--check`/`--fetch` never mutate the skill. After editing the
reference from freshly fetched pages, run `--sync` so the next `--check` is clean.
