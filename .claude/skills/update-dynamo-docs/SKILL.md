---
name: update-dynamo-docs
description: Update an existing page in the Dynamo Fern docs site. Use when editing content, titles, or moving pages between sections.
---

# Update a Dynamo Docs Page

Claude Code skill for updating an existing page in the Dynamo Fern documentation site.

## Related Skills

| Skill | Use When |
|-------|----------|
| [add-dynamo-docs](../add-dynamo-docs/SKILL.md) | Adding a new docs page |
| [rm-dynamo-docs](../rm-dynamo-docs/SKILL.md) | Removing an existing docs page |

---

## Branch Rule

**ALL edits happen on `main` (or a feature branch based on `main`).**
The `docs-website` branch is CI-managed and must **never** be edited by hand.

## Working Directory

Must be in the `dynamo` repo (not `dynamo-tpm`). Architecture details: `docs/README.md`.

## When Invoked

### 1. Locate the Page

Ask for the page to update (accepts any of):
- File path (e.g., `docs/pages/guides/quickstart.md`)
- Page title (e.g., "Quickstart")
- Topic keyword to search for

If given a title or keyword, find the page:
```bash
# Search by title in navigation
grep -n "<title>" docs/versions/dev.yml

# Search by keyword in content
grep -rl "<keyword>" docs/pages/
```

### 2. Read Current Content

Read the page and its navigation entry:
- The markdown file in `docs/pages/`
- The corresponding entry in `docs/versions/dev.yml`

Note the current:
- **Title** (from frontmatter `title:` field)
- **Section** (which sidebar section it belongs to)
- **Path** (relative path in `dev.yml`)

### 3. Apply Edits

Handle three types of changes:

#### Content Only

Edit the markdown file directly. No navigation changes needed.

#### Title Change

1. Update the `title:` field in the markdown frontmatter
2. Update the `- page:` display name in `docs/versions/dev.yml`

#### Section Move

1. Move the markdown file to the new subdirectory:
   ```bash
   git mv docs/pages/<old-subdir>/<file>.md docs/pages/<new-subdir>/<file>.md
   ```
2. Remove the old `- page:` entry from `dev.yml`
3. Add a new `- page:` entry under the target section in `dev.yml`
4. Update the `path:` to reflect the new location

### 4. Check Incoming Links (If Path Changed)

If the file was moved, search for references that need updating:

```bash
# Search for the old path across all docs
grep -r "<old-filename>" docs/pages/ --include="*.md"
grep -r "<old-filename>" docs/versions/
```

Update all references to point to the new path.

### 5. Content Guidelines

Use standard **GitHub-flavored markdown**. For callouts, use GitHub's native syntax — CI auto-converts to Fern format:

```markdown
> [!NOTE]
> Helpful context for the reader.

> [!WARNING]
> Something the reader should be careful about.
```

**Callout mapping** (GitHub → Fern):

| GitHub Syntax | Fern Component |
|---|---|
| `> [!NOTE]` | `<Note>` |
| `> [!TIP]` | `<Tip>` |
| `> [!IMPORTANT]` | `<Info>` |
| `> [!WARNING]` | `<Warning>` |
| `> [!CAUTION]` | `<Error>` |

### 6. Validate

```bash
cd docs
ln -sf . fern  # symlink required by Fern CLI
fern check
fern docs broken-links
```

### 7. Preview Locally (Optional)

```bash
cd docs
fern docs dev
```

Opens a local preview at `http://localhost:3000` with hot reload. No token required.

### 8. Commit

```bash
git add docs/pages/ docs/versions/dev.yml
git commit -s -m "docs: update <page-title>"
```

## Debugging

### `fern check` fails

- **Invalid YAML in `dev.yml`:** Check indentation — nav entries use 2-space indent. A `- page:` must be inside a `contents:` block.
- **Missing file:** If you moved a page, the `path:` in `dev.yml` must match the new location.
- **Duplicate entry:** The same page appears twice in `dev.yml`. Remove the old entry after a section move.

### `fern docs broken-links` reports errors

- **Stale internal links:** After moving or renaming a page, other pages may still link to the old path. Search with `grep -r "<old-filename>" docs/pages/` and update references.
- **Anchor not found:** A `#section-heading` link doesn't match any heading in the target page. Check if the heading text changed.

### CI fails after merge

- **MDX parse error:** Angle-bracket URLs like `<https://example.com>` break MDX parsing. Use `[text](https://example.com)` instead.
- **Broken links check:** The `detect_broken_links.py` job found stale references to the old path. Fix all incoming links before merging.
- **Fern publish error:** Check the Actions tab for the `Fern Docs` workflow. Common causes: expired `FERN_TOKEN`, invalid `docs.yml` syntax, or a moved file that wasn't synced to `docs-website`.

### Changes don't appear on the live site

- **Title mismatch:** You updated the frontmatter `title:` but not the `- page:` name in `dev.yml` (or vice versa). Keep them in sync.
- **Sync delay:** After merge to `main`, the sync-dev workflow takes a few minutes to publish.

## Key References

| File | Purpose |
|------|---------|
| `docs/versions/dev.yml` | Navigation tree — update entries here if title/path changes |
| `docs/pages/` | Content directory — edit pages here |
| `docs/assets/` | Images, SVGs, fonts |
| `docs/convert_callouts.py` | Callout conversion rules (GitHub → Fern) |
| `docs/README.md` | Full architecture guide |
