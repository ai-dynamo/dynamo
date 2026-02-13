---
title: Building and Publishing Docs
---

# How the Fern Documentation Website Works

This document describes the architecture, workflows, and maintenance procedures for the
NVIDIA Dynamo documentation website powered by [Fern](https://buildwithfern.com).

> [!NOTE]
> The documentation website is hosted entirely on
> [Fern](https://buildwithfern.com). CI publishes to
> `dynamo.docs.buildwithfern.com`; the production domain
> `docs.dynamo.nvidia.com` is a custom domain alias that points to the
> Fern-hosted site. There is no separate server — Fern handles hosting,
> CDN, and versioned URL routing.

**Live URLs:**

| Environment | URL |
|---|---|
| Fern-hosted (primary) | <https://dynamo.docs.buildwithfern.com/dynamo> |
| Custom domain (alias) | <https://docs.dynamo.nvidia.com/dynamo> |

---

## Table of Contents

- [Branch Architecture](#branch-architecture)
- [Directory Layout](#directory-layout)
- [Configuration Files](#configuration-files)
- [GitHub Workflows](#github-workflows)
  - [Fern Docs Workflow](#fern-docs-workflow-fern-docsyml)
  - [Docs Link Check Workflow](#docs-link-check-workflow-docs-link-checkyml)
- [Content Authoring](#content-authoring)
- [Callout Conversion](#callout-conversion)
- [Running Locally](#running-locally)
- [Version Management](#version-management)
- [How Publishing Works](#how-publishing-works)
- [Common Tasks](#common-tasks)

---

## Branch Architecture

The documentation system uses a **dual-branch model**:

| Branch | Purpose | Docs directory |
|---|---|---|
| `main` | Source of truth for **dev** (unreleased) documentation | `docs/` |
| `docs-website` | Published documentation including **all versioned snapshots** | `docs/` |

Authors edit pages on `main`. A GitHub Actions workflow automatically syncs
changes to the `docs-website` branch and publishes them to Fern. The
`docs-website` branch is never edited by hand — it is entirely managed by CI.

### Why two branches?

The `docs-website` branch accumulates versioned snapshots over time (e.g.
`pages-v0.8.0/`, `pages-v0.8.1/`). Keeping these on a separate branch avoids
bloating the `main` branch with frozen copies of old documentation.

---

## Directory Layout

### On `main`

```
docs/
├── fern.config.json          # Fern org + CLI version pin
├── docs.yml                  # Site configuration (instances, branding, layout)
├── versions/
│   └── dev.yml               # Navigation tree for the dev version
├── pages/                    # Markdown content (the actual docs)
│   ├── getting-started/
│   ├── guides/
│   ├── kubernetes/
│   ├── reference/
│   └── ...
├── assets/                   # Images, fonts, SVGs, logos
├── components/
│   └── CustomFooter.tsx      # React component for the site footer
├── main.css                  # Custom CSS (NVIDIA branding, dark mode, etc.)
├── convert_callouts.py       # GitHub → Fern admonition converter script
└── diagrams/                 # D2 diagram source files
```

### On `docs-website`

The `docs-website` branch mirrors the above structure, plus versioned snapshots:

```
docs/
├── docs.yml                  # Includes the full versions array
├── versions/
│   ├── dev.yml               # "Next" / dev navigation (synced from main)
│   ├── v0.8.1.yml            # Navigation for v0.8.1 snapshot
│   └── v0.8.0.yml            # Navigation for v0.8.0 snapshot
├── pages/                    # Current dev content (synced from main)
├── pages-v0.8.1/             # Frozen snapshot of pages/ at v0.8.1
├── pages-v0.8.0/             # Frozen snapshot of pages/ at v0.8.0
└── ...                       # (other files same as main)
```

Each `pages-vX.Y.Z/` directory is an immutable copy of `pages/` taken at
release time. The corresponding `versions/vX.Y.Z.yml` file is a copy of
`dev.yml` with all `../pages/` paths rewritten to `../pages-vX.Y.Z/`.

---

## Configuration Files

### `fern.config.json`

```json
{
  "organization": "nvidia",
  "version": "3.73.0"
}
```

- **organization**: The Fern organization that owns the docs site.
- **version**: Pins the Fern CLI version used for generation.

### `docs.yml`

This is the main Fern site configuration. Key sections:

| Section | Purpose |
|---|---|
| `instances` | Deployment targets — staging URL and custom production domain |
| `products` | Defines the product ("Dynamo") and its version list |
| `navbar-links` | GitHub repo link in the navigation bar |
| `footer` | Points to `CustomFooter.tsx` React component |
| `layout` | Page width, sidebar width, searchbar placement, etc. |
| `colors` | NVIDIA green (`#76B900`) accent, black/white backgrounds |
| `typography` | NVIDIA Sans body font, Roboto Mono code font |
| `logo` | NVIDIA logos (dark + light variants), 20px height |
| `js` | Adobe Analytics script injection |
| `css` | Custom `main.css` stylesheet |

**Important:** On `main`, `docs.yml` only lists the `dev` version. On
`docs-website`, it contains the **full versions array** (dev + all releases).
The sync workflow preserves the versions array from `docs-website` when copying
`docs.yml` from `main`.

### `versions/dev.yml`

Defines the navigation tree — the sidebar structure of the docs site. Each
entry maps a page title to a markdown file path:

```yaml
navigation:
  - section: Getting Started
    contents:
      - page: Quickstart
        path: ../pages/getting-started/quickstart.md
      - page: Support Matrix
        path: ../pages/reference/support-matrix.md
```

Sections can be nested. Pages can be marked as `hidden: true` to make them
accessible by URL but invisible in the sidebar.

---

## GitHub Workflows

### Fern Docs Workflow (`fern-docs.yml`)

**Location:** `.github/workflows/fern-docs.yml`

This single consolidated workflow handles linting, syncing, versioning, and
publishing. It runs three jobs depending on the trigger:

#### Job 1: Lint (PRs)

**Triggers:** Pull requests that modify `docs/**` files.

**Steps:**
1. `fern check` — validates Fern configuration syntax
2. `fern docs broken-links` — checks for broken internal links

**Purpose:** Catches broken docs before they merge.

#### Job 2: Sync dev (push to `main`)

**Triggers:** Push to `main` that modifies `docs/**` files, or manual
`workflow_dispatch` (with no tag specified).

**Steps:**
1. Checks out both `main` and `docs-website` branches side-by-side
2. Copies from `main` → `docs-website`:
   - `docs/pages/` — all markdown content
   - `docs/versions/dev.yml` — navigation structure
   - `docs/assets/` — images, fonts, SVGs
   - `docs/fern.config.json` — Fern config
   - `docs/components/` — React components
   - `docs/main.css` — custom styles
   - `docs/convert_callouts.py` — conversion script
3. Runs `convert_callouts.py` to transform GitHub-style callouts to Fern format
4. Updates `docs.yml` from `main` **while preserving the versions array** from
   `docs-website` (uses `yq` to save/restore the versions list)
5. Commits and pushes to `docs-website`
6. Publishes to Fern via `fern generate --docs`

**Symlink trick:** The Fern CLI expects a `fern/` directory. Since docs live in
`docs/`, the workflow creates a symlink `docs/fern → docs/.` (i.e., pointing to
itself) so Fern can find its config files.

#### Job 3: Version Release (tags)

**Triggers:** New Git tags matching `vX.Y.Z` (e.g., `v0.9.0`, `v1.0.0`), or
manual `workflow_dispatch` with a tag specified.

**Steps:**
1. Validates tag format (must be exactly `vX.Y.Z`, no suffixes like `-rc1`)
2. Checks that the version doesn't already exist (no duplicate snapshots)
3. Creates `docs/pages-vX.Y.Z/` by copying `docs/pages/`
4. Rewrites GitHub links in the snapshot:
   - `github.com/ai-dynamo/dynamo/tree/main` → `tree/vX.Y.Z`
   - `github.com/ai-dynamo/dynamo/blob/main` → `blob/vX.Y.Z`
5. Runs `convert_callouts.py` on the snapshot
6. Creates `docs/versions/vX.Y.Z.yml` from `dev.yml` with paths updated to
   `../pages-vX.Y.Z/`
7. Updates `docs.yml`:
   - Inserts new version right after the "dev" entry
   - Sets the product's default `path` to the new version
   - Updates the "Latest" display-name to `"Latest (vX.Y.Z)"`
8. Commits and pushes to `docs-website`
9. Publishes to Fern via `fern generate --docs`

**Anti-recursion note:** Pushes made with `GITHUB_TOKEN` do not trigger other
workflows (GitHub's built-in guard). This is why the publish step is inline in
each job rather than in a separate workflow.

### Docs Link Check Workflow (`docs-link-check.yml`)

**Location:** `.github/workflows/docs-link-check.yml`

**Triggers:** Push to `main` and pull requests.

Runs two independent link-checking jobs:

| Job | Tool | What it checks |
|---|---|---|
| `lychee` | [Lychee](https://lychee.cli.rs/) | External HTTP links (with caching, retries, rate-limit handling). Runs offline for PRs. |
| `broken-links-check` | Custom Python script (`detect_broken_links.py`) | Internal relative markdown links and symlinks. Creates GitHub annotations on PRs pointing to exact lines with broken links. |

---

## Content Authoring

### Writing docs on `main`

1. Edit or add markdown files in `docs/pages/`.
2. If adding a new page, add an entry in `docs/versions/dev.yml` to make it
   appear in the sidebar navigation.
3. Use standard GitHub-flavored markdown. Callouts (admonitions) should use
   GitHub's native syntax — they are automatically converted during sync:
   ```markdown
   > [!NOTE]
   > This is a note that will become a Fern `<Note>` component.

   > [!WARNING]
   > This warning will become a Fern `<Warning>` component.
   ```
4. Open a PR. The lint jobs (`fern check`, `fern docs broken-links`, lychee,
   broken-links-check) run automatically.
5. Once merged to `main`, the sync-dev workflow publishes changes within minutes.

### Assets and images

Place images in `docs/assets/` and reference them with relative paths from your
markdown files:

```markdown
![Architecture diagram](../assets/architecture.png)
```

### Custom components

React components in `docs/components/` can be used in markdown via MDX. The
`CustomFooter.tsx` renders the NVIDIA footer with legal links and branding.

---

## Callout Conversion

The `docs/convert_callouts.py` script bridges the gap between GitHub-flavored
markdown and Fern's admonition format. This lets authors use GitHub's native
callout syntax on `main` while Fern gets its required component format.

### Mapping

| GitHub Syntax | Fern Component |
|---|---|
| `> [!NOTE]` | `<Note>` |
| `> [!TIP]` | `<Tip>` |
| `> [!IMPORTANT]` | `<Info>` |
| `> [!WARNING]` | `<Warning>` |
| `> [!CAUTION]` | `<Error>` |

### Usage

```bash
# Convert all files in a directory (recursive, in-place)
python convert_callouts.py --dir docs/pages

# Convert a single file
python convert_callouts.py input.md output.md

# Run built-in tests
python convert_callouts.py --test
```

The conversion happens automatically during the sync-dev and release-version
workflows. Authors never need to run it manually.

---

## Running Locally

You can preview the documentation site on your machine using the
[Fern CLI](https://buildwithfern.com/learn/cli-api/overview). This is useful
for verifying layout, navigation, and content before opening a PR.

### Prerequisites

Install the Fern CLI globally via npm:

```bash
npm install -g fern-api
```

### Create the `fern` symlink

The Fern CLI requires its configuration files to live inside a directory called
`fern/`. In this repo the docs live in `docs/`, so you need to create a symlink
that points `fern` back to the same directory:

```bash
cd docs
ln -s . fern
```

This makes the CLI find `fern/fern.config.json`, `fern/docs.yml`, etc. without
moving any files. The symlink is listed in `.gitignore` and should not be
committed.

### Validate configuration

Run `fern check` to validate that `docs.yml`, `fern.config.json`, and the
navigation files are syntactically correct:

```bash
cd docs
fern check
```

### Check for broken links

Use `fern docs broken-links` to scan all pages for internal links that don't
resolve:

```bash
cd docs
fern docs broken-links
```

This is the same check that runs in CI on every pull request.

### Start a local preview server

Run `fern docs dev` to build the site and serve it locally with hot-reload:

```bash
cd docs
fern docs dev
```

> [!NOTE]
> `fern docs dev` requires a valid `FERN_TOKEN` environment variable. Ask a
> maintainer for access, or set it in your shell profile:
> ```bash
> export FERN_TOKEN=<your-token>
> ```

The local server lets you see exactly how pages will look on the live site,
including navigation, version dropdowns, and custom styling.

---

## Version Management

### How versions work

The Fern site supports a version dropdown in the UI. Each version is defined by:

1. **A navigation file** (`docs/versions/vX.Y.Z.yml`) — sidebar structure
   pointing to version-specific pages.
2. **A pages directory** (`docs/pages-vX.Y.Z/`) — frozen snapshot of the
   markdown content at release time.
3. **An entry in `docs.yml`** — tells Fern about the version's display name,
   slug, and config path.

### Version types

| Version | Display Name | Slug | Description |
|---|---|---|---|
| Latest | `Latest (vX.Y.Z)` | `/` | Default version; points to the newest release |
| Stable releases | `vX.Y.Z` | `vX.Y.Z` | Immutable snapshots |
| Dev | `dev` | `dev` | Tracks `main`; updated on every push |

### URL structure

- **Latest (default):** `docs.dynamo.nvidia.com/dynamo/`
- **Specific version:** `docs.dynamo.nvidia.com/dynamo/v0.8.1/`
- **Dev:** `docs.dynamo.nvidia.com/dynamo/dev/`

### Creating a new version

Simply push a semver tag:

```bash
git tag v0.9.0
git push origin v0.9.0
```

The `release-version` job in `fern-docs.yml` handles everything else
automatically.

---

## How Publishing Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTINUOUS (dev)                             │
│                                                                     │
│  Developer pushes to main                                           │
│       │                                                             │
│       ▼                                                             │
│  docs/** changed? ── No ──▶ (nothing happens)                      │
│       │                                                             │
│      Yes                                                            │
│       │                                                             │
│       ▼                                                             │
│  sync-dev job:                                                      │
│    1. Copy docs/pages/, assets/, configs → docs-website branch      │
│    2. Convert GitHub callouts → Fern admonitions                    │
│    3. Preserve version list from docs-website's docs.yml            │
│    4. Commit + push to docs-website                                 │
│    5. fern generate --docs (publishes to Fern)                      │
│       │                                                             │
│       ▼                                                             │
│  Live on docs.dynamo.nvidia.com/dynamo/dev/ within minutes          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      VERSION RELEASE                                │
│                                                                     │
│  Maintainer pushes vX.Y.Z tag                                       │
│       │                                                             │
│       ▼                                                             │
│  release-version job:                                               │
│    1. Validate tag format (vX.Y.Z only)                             │
│    2. Check version doesn't already exist                           │
│    3. Snapshot pages/ → pages-vX.Y.Z/                               │
│    4. Rewrite GitHub links (tree/main → tree/vX.Y.Z)               │
│    5. Convert callouts in snapshot                                  │
│    6. Create versions/vX.Y.Z.yml (paths → pages-vX.Y.Z/)          │
│    7. Update docs.yml (insert version, set as default)              │
│    8. Commit + push to docs-website                                 │
│    9. fern generate --docs (publishes to Fern)                      │
│       │                                                             │
│       ▼                                                             │
│  New version visible in dropdown at docs.dynamo.nvidia.com/dynamo/  │
└─────────────────────────────────────────────────────────────────────┘
```

### Secrets

| Secret | Purpose |
|---|---|
| `FERN_TOKEN` | Authentication token for `fern generate --docs`. Required for publishing. Stored in GitHub repo secrets. |

---

## Common Tasks

### Update existing documentation

1. Edit files in `docs/pages/` on a feature branch.
2. If adding a new page, add its entry in `docs/versions/dev.yml`.
3. Open a PR — linting runs automatically.
4. Merge — sync + publish happens automatically.

### Add a new top-level section

1. Create a directory under `docs/pages/` (e.g., `docs/pages/new-section/`).
2. Add markdown files for each page.
3. Add a new `- section:` block in `docs/versions/dev.yml` with the desired
   hierarchy.

### Release versioned documentation

```bash
git tag v1.0.0
git push origin v1.0.0
```

That's it. The workflow snapshots the current dev docs, creates the version
config, and publishes.

### Manually trigger a sync or release

Go to **Actions → Fern Docs → Run workflow**:
- Leave **tag** empty to trigger a dev sync.
- Enter a tag (e.g., `v0.9.0`) to trigger a version release.

### Debug a failed publish

1. Check the **Actions** tab for the failed `Fern Docs` workflow run.
2. Common issues:
   - **Broken links:** Fix the links flagged by `fern docs broken-links`.
   - **Invalid YAML:** Check `docs.yml` or `versions/dev.yml` syntax.
   - **Expired `FERN_TOKEN`:** Rotate the token in repo secrets.
   - **Duplicate version:** The tag was already released; check `docs-website`
     for existing `pages-vX.Y.Z/` directory.
