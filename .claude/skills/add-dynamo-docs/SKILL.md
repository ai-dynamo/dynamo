---
name: add-dynamo-docs
description: Add a new page to the Dynamo Fern docs site. Use when creating new documentation pages.
---

# Add a Dynamo Docs Page

Claude Code skill for adding a new page to the Dynamo Fern documentation site.

## Related Skills

| Skill | Use When |
|-------|----------|
| [rm-dynamo-docs](../rm-dynamo-docs/SKILL.md) | Removing an existing docs page |
| [update-dynamo-docs](../update-dynamo-docs/SKILL.md) | Editing an existing docs page |

---

## Branch Rule

**ALL edits happen on `main` (or a feature branch based on `main`).**
The `docs-website` branch is CI-managed and must **never** be edited by hand.

## Working Directory

Must be in the `dynamo` repo (not `dynamo-tpm`). Architecture details: `docs/README.md`.

## When Invoked

### 1. Gather Information

Ask for:
- **Page title** — appears in the sidebar and as the H1
- **Target section** — which sidebar section (e.g., `Getting Started`, `User Guides`, `Components`)
- **Filename** — kebab-case `.md` file (e.g., `my-new-feature.md`)
- **Subdirectory** — which `docs/` subdirectory (e.g., `getting-started`, `features`, `components`)

### 2. Create the Page

Create `docs/<subdirectory>/<filename>.md` with SPDX header and Fern frontmatter:

```markdown
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: <Page Title>
---

# <Page Title>

<!-- Content goes here -->
```

### 3. Add Navigation Entry

Edit `docs/index.yml` and add the page under the correct section:

```yaml
- page: <Page Title>
  path: <subdirectory>/<filename>.md
```

**Section locations in `index.yml`** (search for the comment banner):
- `# ==================== Getting Started ====================`
- `# ==================== Kubernetes Deployment ====================`
- `# ==================== User Guides ====================`
- `# ==================== Backends ====================`
- `# ==================== Components ====================`
- `# ==================== Integrations ====================`
- `# ==================== Documentation ====================`
- `# ==================== Design Docs ====================`
- `# ==================== Blog ====================`
- `# ==================== Hidden Pages ====================`

### 4. Write Content

Use standard **GitHub-flavored markdown**. For callouts, use GitHub's native syntax — CI auto-converts to Fern format:

```markdown
> [!NOTE]
> Helpful context for the reader.

> [!WARNING]
> Something the reader should be careful about.

> [!TIP]
> A useful suggestion.
```

**Callout mapping** (GitHub → Fern):

| GitHub Syntax | Fern Component |
|---|---|
| `> [!NOTE]` | `<Note>` |
| `> [!TIP]` | `<Tip>` |
| `> [!IMPORTANT]` | `<Info>` |
| `> [!WARNING]` | `<Warning>` |
| `> [!CAUTION]` | `<Error>` |

Reference images from `docs/assets/`:
```markdown
![Diagram](../assets/my-diagram.png)
```

### 5. Validate

```bash
fern check
fern docs broken-links
```

### 6. Preview Locally (Optional)

```bash
fern docs dev
```

Opens a local preview at `http://localhost:3000` with hot reload. No token required.

### 7. Commit

```bash
git add docs/<subdirectory>/<filename>.md docs/index.yml

git commit -s -m "docs: add <page-title> page"
```

## Debugging

### `fern check` fails

- **Invalid YAML in `index.yml`:** Check indentation — nav entries use 2-space indent. A `- page:` must be inside a `contents:` block.
- **Missing file:** The `path:` in `index.yml` must match the actual file location. Paths are relative to `docs/` (e.g., `getting-started/quickstart.md`).

### `fern docs broken-links` reports errors

- **Broken internal link:** A markdown link reference points to a file that doesn't exist. Fix the path or remove the link.
- **Anchor not found:** A `#section-heading` link doesn't match any heading in the target page.

### CI fails after merge

- **MDX parse error:** Angle-bracket URLs like `<https://example.com>` break MDX parsing. Use `[text](https://example.com)` instead.
- **Broken links check:** The `detect_broken_links.py` job checks relative links across all docs. If your new page links to a file that doesn't exist yet, CI will fail.
- **Fern publish error:** Check the Actions tab for the `Fern Docs` workflow. Common causes: expired `FERN_TOKEN`, invalid `fern/docs.yml` syntax, or a file referenced in `index.yml` that wasn't synced to `docs-website`.

### Page doesn't appear on the live site

- **Missing nav entry:** The page exists but isn't in `docs/index.yml`. Add it.
- **Hidden section:** The page is inside a `hidden: true` section. It's accessible by direct URL but won't appear in the sidebar.
- **Sync delay:** After merge to `main`, the sync-dev workflow takes a few minutes to publish.

## Key References

| File | Purpose |
|------|---------|
| `docs/index.yml` | Navigation tree — add entries here |
| `docs/` | Content directory — create pages here |
| `docs/assets/` | Images, SVGs, fonts |
| `fern/docs.yml` | Fern site configuration |
| `fern/convert_callouts.py` | Callout conversion rules (GitHub → Fern) |
| `docs/README.md` | Full architecture guide |
