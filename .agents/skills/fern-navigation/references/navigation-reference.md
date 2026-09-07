<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Fern Navigation & Site-Structure Reference

Complete field-level reference for Fern's site-level navigation and configuration. Distilled from
`fern-api/docs` @ `main`, `fern/products/docs/pages/navigation/*.mdx` (see `manifest.json` for the
synced SHAs). Sections below map 1:1 to the eight source pages.

> **Plans:** Versions and Products are **Team/Enterprise** features. Everything else is available on
> all plans.

> **Paths are relative to the YAML file they're set in** (`docs.yml`, a product `.yml`, or a version
> `.yml`), unless noted otherwise (some frontmatter fields require absolute URLs).

---

## 0. Two levels of configuration

- **Site-level** → `docs.yml`: appearance, structure, behavior for the whole site.
- **Page-level** → frontmatter in each `.md`/`.mdx` file: overrides for one page.

Enable editor validation/autocomplete by adding a schema directive at the top of the file:

```yaml docs.yml
# yaml-language-server: $schema=https://schema.buildwithfern.dev/docs-yml.json
```
```yaml products/product-a.yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/fern-api/fern/main/product-yml.schema.json
```

---

## 1. Sections, pages, and folders (`navigation:`)

The `navigation` key in `docs.yml` defines the sidebar. It's a list of **sections**, **pages**,
**folders**, **links**, `api:` references, and `changelog:` references.

### Sections

```yaml docs.yml
navigation:
  - section: Introduction
    contents:
      - page: My page
        path: ./pages/my-page.mdx
      - page: Another page
        path: ./pages/another-page.mdx
```

Sections **nest** for multi-level hierarchies (a section's `contents` can contain more sections).
Add an **overview page** for a section with a sibling `path:` on the section:

```yaml
  - section: Guides
    path: ./pages/guide-overview.mdx   # section landing page
    contents:
      - page: Simple guide
        path: ./pages/guides/simple.mdx
```

Sections also support `slug` / `skip-slug` (URL control) and `icon`.

### Pages

```yaml
- page: My page          # display + sidebar title (unless frontmatter overrides)
  path: ./pages/my-page.mdx
```

`.md` or `.mdx`. See the .md/.mdx rule in **dynamo-docs** for this repo.

### Folders (auto-discovery)

```yaml
- folder: ./pages/guides
  title: Guides              # optional, defaults to folder name
  slug: user-guides          # optional custom URL segment
  skip-slug: false           # optional; omit folder from URL path
  title-source: frontmatter  # 'filename' (default) | 'frontmatter'
  collapsed: true            # optional
```

For a folder, Fern automatically:
- derives titles and slugs from filenames,
- creates nested sections from subdirectories,
- sorts pages alphabetically,
- uses `index.md`/`index.mdx` (case-insensitive) as the section overview.

Per-page **`position:`** frontmatter (a number) forces ordering within a folder (positioned pages
first, numerically; then the rest alphabetically). `title-source` here overrides the global
`settings.folder-title-source`.

### Slugs and URL paths

Fern builds a URL by combining slugs from every level (product, version, tab, section, folder, page).
Each level auto-slugs from its display name (or filename for folders).

```yaml
navigation:
  - section: Get Started      # slug renamed to "start"
    slug: start
    contents:
      - section: Guides       # skipped — omitted from URL
        skip-slug: true
        contents:
          - page: Quickstart   # slug renamed to "quick"
            slug: quick
            path: ./pages/quickstart.mdx
```

→ page hosted at `/start/quick`. (Frontmatter `slug:` on a page overrides the section/folder portion
while preserving product/version prefix.)

### Hiding content

`hidden: true` on a page, folder, or section: still reachable by direct URL, but excluded from search
and indexing. Applies to all pages inside a hidden folder/section.

### Availability badges

Set `availability:` on a page/section/folder to render a sidebar badge without hardcoding it in the
title:

| Value | Sidebar badge |
|---|---|
| `beta` | Yes |
| `pre-release` | Yes |
| `in-development` | Yes |
| `deprecated` | Yes — title struck through |
| `stable` | No (default) |
| `generally-available` | No (default) |

Children inherit from parent unless overridden by a per-page `availability` in `docs.yml`, or by
**page frontmatter** `availability` (which beats all `docs.yml` values). For versioned sites, set
these in the version `.yml` files. API Reference sections/endpoints render the same badges.

### Collapsed sections/folders

By default sections/folders are expanded and **not** collapsible. The `collapsed:` property:

| Value | Behavior |
|---|---|
| `true` | Collapsed on load; user can expand. |
| `open-by-default` | Expanded on load, but collapsible (shows a toggle). |

### Sidebar icons

`icon:` on sections/pages/folders (and `api:`). Three formats — Font Awesome class
(`fa-regular fa-home`), custom image path (`./assets/icons/intro.svg`), or inline SVG string.

### External links in the sidebar

```yaml
- link: Our YouTube channel
  href: https://www.youtube.com/
  # target: _blank   # optional
```

### API Reference section

The special `api:` key generates an API Reference section:

```yaml
navigation:
  - section: Introduction
    contents: [...]
  - api: API Reference
    icon: fa-regular fa-puzzle   # optional
```

---

## 2. Tabs and tab variants

Declare `tabs:` at the top level of `docs.yml`, then reference each in `navigation:` (with `layout`,
`variants`, or `href`).

```yaml docs.yml
tabs:
  api:
    display-name: API Reference
    icon: puzzle
  help:
    display-name: Help center
    icon: ./assets/icons/help-icon.svg
  github:
    display-name: GitHub
    icon: brands github
    href: https://github.com/fern-api/fern
    target: _blank

navigation:
  - tab: api
    layout:
      - section: Introduction
        contents:
          - page: My page
            path: my-page.mdx
      - api: API Reference
  - tab: help
    layout:
      - section: Help center
        contents:
          - page: Contact us
            path: contact-us.mdx
  - tab: github          # href tab — no layout/variants
```

### Tab properties

| Field | Type | Notes |
|---|---|---|
| `display-name` | string (**required**) | Tab header text. |
| `icon` | string | FA class, image path, or inline SVG. |
| `slug` | string | Custom URL slug. |
| `skip-slug` | boolean | Exclude tab slug from URLs. |
| `hidden` | boolean | Hide from nav (direct URL still works). |
| `layout` | list | Nav structure for the tab. Required unless `variants`/`href`. |
| `variants` | list | Tab variants (below). Use instead of `layout`. |
| `href` | string | External URL. Must NOT combine with `layout`/`variants`. |
| `target` | string | `_blank` \| `_self` \| `_parent` \| `_top`. |
| `changelog` | string | Path to a changelog folder (relative to the YAML file). |
| `viewers` | string \| list | Role-based access control. |
| `orphaned` | boolean | When true, roles don't inherit from parents. |
| `feature-flag` | string \| object | Conditional display. |

**Common errors:** tab referenced in `navigation` but not declared in `tabs:`; tab with both `href`
and `layout`/`variants`; tab with none of `href`/`layout`/`variants`.

### Placement & style

Via `theme.tabs` (or `layout.tabs-placement`):

```yaml docs.yml
theme:
  tabs:
    style: bubble        # "default" (underline) | "bubble" (pill)
    alignment: center    # "left" | "center" (center applies only to header tabs)
    placement: header    # "header" | "sidebar" (default: sidebar)
```

`theme.tabs.placement` takes precedence over `layout.tabs-placement`. Ignored when
`layout.disable-header: true`.

### Tab variants

Different content perspectives within one tab (dev vs PM, REST vs GraphQL). Supports RBAC. Use
`variants:` instead of `layout:`:

```yaml
navigation:
  - tab: help
    variants:
      - title: For developers
        layout:
          - section: Getting started
            contents:
              - page: Quick start
                path: ./pages/dev-quickstart.mdx
      - title: For product managers
        layout:
          - section: Getting started
            contents:
              - page: Overview
                path: ./pages/pm-overview.mdx
```

Variant properties: `title` (**required**), `layout` (**required**), `subtitle`, `icon`, `slug`,
`skip-slug`, `hidden`, `default` (first variant is default if unset), `viewers`, `orphaned`,
`feature-flag`.

> **When variants vs tabs:** variants = different perspectives on the *same* area; tabs = *different*
> documentation sections.

---

## 3. Versions (Team plan)

Each version can have its own tabs/sections/pages/API refs, and can share content. For in-page
version-conditional content use the `<Versions>` component (fern-components); site-wide versioning and
`<Versions>` are independent but combinable.

### File layout

```
fern/
  ├─ docs.yml
  ├─ pages/...
  └─ versions/
     ├─ latest.yml
     ├─ latest/pages/...
     ├─ v2.yml
     └─ v2/pages/...
```

Each version `.yml` holds that version's `navigation` (and `tabs`, if any).

### Register in `docs.yml`

```yaml docs.yml
versions:
  - display-name: Latest          # shown in the dropdown
    path: ./versions/latest.yml
    availability: beta            # optional
  - display-name: V2
    path: ./versions/v2.yml
    slug: v2                      # optional; else auto from display-name
    availability: stable
```

- The **first** version is the default (unversioned paths); others get versioned paths (`/v2/...`).
- **Remove top-level `navigation`/`tabs` from `docs.yml`** — they belong in the version files.

### Options

- **`availability`**: `deprecated` | `ga` | `stable` | `beta` (badge in the dropdown).
- **`slug`**: override the URL segment (default = lowercased, hyphenated `display-name`).
- **`audiences`**: gate a version to matching `instances[].audiences` (below). No tag → shown by
  default; non-match → excluded.
- **`hidden: true`**: keep reachable by direct URL, out of nav/search/index. The default (first)
  version **can't** be hidden.
- **Conditional content**: `<If versions={["v2"]}>…</If>`.
- **Search scoping**: `settings.search` (boost or default-filter by current version).

### Styling

`.fern-version-selector` (the control) and `.fern-version-selector-radio-group` (the dropdown).

---

## 4. Products (Team plan)

A product switcher for multi-product sites. Each product has its own navigation, tabs, versions, and
API refs; products can share content. **Internal** (hosted here) or **external** (link out).

### File layout

```
fern/
  ├─ docs.yml
  ├─ pages/
  └─ products/
     ├─ product-a.yml           # unversioned product
     └─ product-b/              # versioned product
        ├─ product-b.yml
        └─ versions/
           ├─ latest/latest.yml + pages/
           └─ v2/v2.yml + pages/
```

Each internal product `.yml` holds its `navigation` (and `tabs`). Example:

```yaml products/product-a.yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/fern-api/fern/main/product-yml.schema.json
navigation:
  - section: Introduction
    contents:
      - page: Shared Resource
        path: ../pages/shared-resource.mdx   # products can share content
  - api: API Reference
```

### Register in `docs.yml`

```yaml docs.yml
products:
  - display-name: Product A
    path: ./products/product-a.yml
    icon: fa-solid fa-leaf        # optional (FA / image path / inline SVG)
    slug: product-a               # optional (internal only)
    subtitle: Product A subtitle  # optional
  - display-name: Product B
    path: ./products/product-b/versions/latest/latest.yml  # default = latest
    image: ./images/product-b.png # optional (image beats icon if both set)
    versions:                     # optional (internal only)
      - display-name: Latest
        path: ./products/product-b/versions/latest/latest.yml
      - display-name: V2
        path: ./products/product-b/versions/v2/v2.yml
        availability: stable
  - display-name: Dashboard       # external product
    href: https://dashboard.example.com
    icon: "<svg ...>...</svg>"
    subtitle: External app
    target: _blank
```

Shared optional params (internal + external): `image`, `icon`, `subtitle`. Internal-only: `slug`,
`versions`. External needs `href` (+ optional `target`) and supports **no** `navigation`/`tabs`.

### Hard constraint

**Remove top-level `navigation`/`tabs` from `docs.yml`** when using products. A top-level
`navigation` block can't coexist with `products:` — `fern check` rejects it and it would render an
empty site. For **versioned** products, also move `navigation`/`tabs` out of the product `.yml` into
the version `.yml` files.

### Landing page (root, product-independent)

```yaml docs.yml
landing-page:
  page: Welcome
  path: ./pages/welcome.mdx
  slug: /welcome     # optional
products:
  - display-name: Product A
    path: ./products/product-a.yml
```

### Audiences (instances)

Gate products/versions to instances by audience tag (match → include; non-match → exclude; no tag →
included by default):

```yaml docs.yml
instances:
  - url: internal.docs.buildwithfern.com
    audiences: [internal]
  - url: public.docs.buildwithfern.com
    audiences: [public]
products:
  - display-name: Platform API
    path: ./products/platform-api.yml
    audiences: [public, internal]
    versions:
      - display-name: v3
        path: ./versions/v3.yml
        audiences: [public]
      - display-name: v2 (Internal)
        path: ./versions/v2.yml
        audiences: [internal]
  - display-name: Admin Tools
    path: ./products/admin-tools.yml
    audiences: [internal]
```

Composes with API Reference audiences (which filter endpoints/schemas within a product).

### Conditional content & search

- `<If products={["orchids"]}>…</If>` — render per current product.
- `settings.search` — boost/default-filter results by current product.

### Selector styling

- `.fern-product-selector` / `.fern-version-selector` — the controls.
- `.fern-product-selector-radio-group` / `.fern-version-selector-radio-group` — the dropdowns
  (e.g. `display: grid; grid-template-columns: repeat(2, 1fr);` for a grid layout).
- `theme.product-switcher: default | toggle` — dropdown vs horizontal toggle bar.

---

## 5. Changelog pages

Renders a scannable, searchable, tag-filterable timeline of dated entry cards. RSS is automatic.

### Folder

A folder named **exactly** `changelog` in your `fern` folder. **No subdirectories** — all entry files
sit flat in its root. Both `.md` and `.mdx` are supported (`.mdx` to use components).

### Reference it in `docs.yml`

As a **tab**:
```yaml docs.yml
tabs:
  changelog:
    display-name: Changelog
    icon: light clock
    changelog: ./changelog
navigation:
  - tab: changelog
```

As a **section**:
```yaml docs.yml
navigation:
  - changelog: ./changelog
    title: Release Notes
    slug: api-release-notes
```
> Section-level changelogs **can't** nest inside an `api` entry.

### Entry files

- Name by date; Fern sorts chronologically. Accepted: `YYYY-MM-DD`, `MM-DD-YYYY`, `MM-DD-YY`.
- Each top-level `##` heading becomes its own card (one dated file → several cards).

### Tags

Filterable badges. Per-file via frontmatter, or per-entry via `<ChangelogTags>`:

```mdx changelog/2024-07-31.mdx
---
tags: ["plants-api", "breaking-change"]
---
## New plant endpoints
<ChangelogTags>plants-api, inventory-management</ChangelogTags>
Added `POST /plant`.
```
Per-entry tags replace the file's frontmatter tags for that card; untagged entries inherit them.
`<ChangelogTags>` accepts comma-separated children or a `tags={[...]}` prop.

### Overview & layout

- An optional `overview.mdx` in the folder renders above the entries.
- Layouts: `timeline` (default, condensed searchable cards) or `classic` (full stacked entries).
  Site-wide via `layout.changelog-layout`; per-changelog override via `layout:` frontmatter in that
  changelog's `overview.mdx`.

### Linking & RSS

- Each entry has a unique URL (e.g. `/changelog/2025/3/31`); search syncs to `?q=`.
- RSS: append `.rss` to the changelog path (e.g. `/docs/changelog.rss`).

---

## 6. Page-level settings (frontmatter)

YAML between `---` fences at the top of a `.md`/`.mdx` file. Values are also processed as MDX, so
some characters need quoting/escaping:

| Characters | Solution | Example |
|---|---|---|
| `:` `#` `&` `*` `!` `%` | wrap in quotes | `title: "OAuth: A guide"` |
| `{` `}` `<` `>` | escape with `\` | `title: "Using \<Callout\>"` |
| `"` `'` | opposite style or `\` | `title: 'The "best" practices'` |

| Field | Type | Purpose |
|---|---|---|
| `title` | string | `<title>` (browser tab, search). Defaults to nav name. Site suffix appended. |
| `sidebar-title` | string | Shorter sidebar label; beats `docs.yml`. |
| `subtitle` | string | Visible under the title; also meta description if `description` unset. |
| `description` | string | Meta description (SEO, llms.txt); not shown on page. |
| `last-updated` | string | "Last updated" footer stamp (free-form date). |
| `slug` | string | Override URL path (keeps product/version prefix). |
| `edit-this-page-url` | string | Absolute GitHub URL for the Edit link (else configure globally). |
| `image` | string | OpenGraph image (absolute URL). |
| `hide-toc` | boolean | Hide right-side table of contents. |
| `max-toc-depth` | number | Deepest heading level in the TOC. |
| `hide-nav-links` | boolean | Hide prev/next footer links. |
| `hide-feedback` | boolean | Hide the feedback form. |
| `hide-page-actions` | boolean | Hide Copy/Markdown/Ask-AI/etc. buttons. |
| `logo` | object | Per-page logo override (`light`/`dark`, absolute URLs). |
| `layout` | string | `guide` (default) · `overview` (wider) · `reference` (full-width, no TOC) · `page` (no TOC/sidebar) · `custom` (blank canvas). |
| `availability` | string | Page badge; beats `docs.yml`. `stable`/`generally-available`/`in-development`/`pre-release`/`deprecated`/`beta`. |
| `position` | number | Ordering within a folder. |
| `tags` | array | Changelog entries only. |
| SEO/OG/Twitter | various | `headline`, `keywords`, `canonical-url`, `og:*`, `twitter:*`, `noindex`, `nofollow`. |

---

## 7. Site-level settings (`docs.yml`) — full surface

### Core top-level keys

`title`, `favicon`, `default-language` (`typescript`/`python`/`java`/`go`/`ruby`/`csharp`/`php`/
`swift`/`curl`), `logo`, `colors` (**required**), `redirects`, `navbar-links`, `background-image`,
`typography`, `layout`, `settings`, `landing-page`, `metadata`, `global-theme`, `header`, `footer`
(the last two = paths to custom React components).

### colors

Only `accent-primary` is required. Each is `{light, dark}`:
`accent-primary`, `background`, `border`, `sidebar-background`, `header-background`, `card-background`.
Exposed as CSS custom properties for custom stylesheets.

### logo

`href`, `dark`, `light`, `right-text`, `height`.

### navbar-links

List; `type:` one of `outlined` | `minimal` | `filled` | `github` | `dropdown`. Fields: `text`,
`href` (or `value` for `github`), `icon`, `rightIcon`, `rounded`, `target`. `dropdown` adds a `links:`
list (`text`/`href`/`icon`/`rightIcon`/`rounded`/`target`).

### footer-links

Social/community URLs: `github`, `slack`, `x`, `twitter`, `linkedin`, `youtube`, `instagram`,
`facebook`, `discord`, `hackernews`, `medium`, `website`.

### typography

`bodyFont` / `headingsFont` / `codeFont`, each with `name` + `path` (single) or `paths` (list of
`{path, weight, style}`); supports variable-font weight ranges (`weight: 400 700`). WOFF2 recommended.

### layout

`header-height`, `page-width` (or `full`), `content-width`, `sidebar-width`,
`searchbar-placement` (`header`/`sidebar`/`header-tabs`), `tabs-placement` (`header`/`sidebar`),
`switcher-placement` (`header`/`sidebar`), `content-alignment` (`center`/`left`),
`disable-header`, `hide-nav-links`, `hide-feedback`, `mobile-toc`,
`changelog-layout` (`timeline`/`classic`).

### theme

`sidebar` (`default`/`minimal`), `body` (`default`/`canvas`), `tabs` (string or `{style, alignment,
placement}`), `page-actions` (`default`/`toolbar`), `footer-nav` (`default`/`minimal`),
`product-switcher` (`default`/`toggle`).

### settings

`search-text`, `disable-search`, `disable-explorer-proxy`, `dark-mode-code`,
`default-search-filters` (legacy alias), `search` (`prioritize-current-product`,
`default-filter-by-current-product`), `http-snippets` (bool or language list), `hide-404-page`,
`use-javascript-as-typescript`, `disable-analytics`, `folder-title-source` (`filename`/`frontmatter`),
`substitute-env-vars` (`${ENV_VAR}` at build time; escape literal as `\$\{VAR\}`).

### page-actions

`default:` (`copy-page`/`view-as-markdown`/`ask-ai`/`chatgpt`/`claude`/`claude-code`/`cursor`/
`vscode`/`install-skills`); `options:` toggles built-ins (`copy-page`, `view-as-markdown`, `ask-ai`,
`chatgpt`, `claude`, `claude-code`, `cursor`, `vscode`, `mcp`) and configures `skills` (Install
skills modal) and `custom` (list of `{title, subtitle, url, icon, default}`; URL placeholders `{slug}`,
`{domain}`, `{url}`).

### instances

List of docs backends: `url` (must end `docs.buildwithfern.com`), `custom-domain`, `edit-this-page`
(`github: {owner, repo, branch}` + `launch: github|dashboard`), `audiences`, `multi-source`.

### redirects

List of `{source, destination}` (exact or regex like `/old/:slug*` → `/new/:slug*`).

### metadata (SEO)

Site-wide OpenGraph/Twitter: `og:site_name`, `og:title`, `og:description`, `og:url`, `og:image`
(+`:width`/`:height`), `og:locale`, `og:logo`, `og:dynamic` (+`:background-image`), `twitter:title`/
`description`/`handle`/`image`/`site`/`card`, `canonical-host`.

### Other sections

- **analytics**: `ga4.measurement-id`, `gtm.container-id`, `posthog.api-key`.
- **integrations**: `context7` (path to verification file → served at `/context7.json`).
- **ai-search (Ask Fern)**: `mask-pii`, `datasources` (`{url, title}`), `system-prompt`.
- **agents**: `page-directive`, `page-description-source` (`description`/`subtitle`), `llms-txt`,
  `robots-txt`.
- **ai-examples**: `enabled`, `style` (≤500 chars).
- **check.rules**: `broken-links`, `example-validation`, `no-non-component-refs`,
  `valid-local-references`, `no-circular-redirects`, `valid-docs-endpoints`, `missing-redirects` —
  each `warn` | `error`.
- **experimental**: `dynamic-snippets` (default true).

---

## Quick recipes

**Add a GitHub link as a nav tab:**
```yaml
tabs: { github: { display-name: GitHub, icon: brands github, href: https://github.com/org/repo, target: _blank } }
navigation: [ { tab: github } ]
```

**Split one site into two products (from a flat `navigation`):**
1. Create `products/a.yml` and `products/b.yml`, each with its own `navigation:` (+ `tabs:`).
2. In `docs.yml`, add a `products:` list pointing at them; **delete** the top-level `navigation:`/`tabs:`.
3. Optionally add a `landing-page:`; add `redirects:` for every moved URL.
4. `fern check`.

**Cut a new version:**
1. Move current content under `versions/latest/` + `versions/latest.yml`; branch `versions/v2/…`.
2. Add a `versions:` list to `docs.yml` (first = default); remove top-level `navigation:`/`tabs:`.
3. Mark old `availability: deprecated` (or `hidden: true`, but not the default).
