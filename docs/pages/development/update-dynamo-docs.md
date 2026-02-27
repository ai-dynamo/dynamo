---
title: "Skill: Update a Docs Page"
hidden: true
---

# Update a Dynamo Docs Page

Claude Code skill for updating an existing page in the Dynamo Fern documentation site.

## Related Skills

| Skill | Use When |
|-------|----------|
| [add-dynamo-docs](add-dynamo-docs.md) | Adding a new docs page |
| [rm-dynamo-docs](rm-dynamo-docs.md) | Removing an existing docs page |
| [fern-website](fern-website.md) | Understanding the docs architecture |

---

## Branch Rule

> [!CAUTION]
> ALL edits happen on `main` (or a feature branch based on `main`).
> The `docs-website` branch is CI-managed and must **never** be edited by hand.

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

### 7. Commit

```bash
git add docs/pages/ docs/versions/dev.yml
git commit -s -m "docs: update <page-title>"
```

## Key References

| File | Purpose |
|------|---------|
| `docs/versions/dev.yml` | Navigation tree — update entries here if title/path changes |
| `docs/pages/` | Content directory — edit pages here |
| `docs/assets/` | Images, SVGs, fonts |
| `docs/convert_callouts.py` | Callout conversion rules (GitHub → Fern) |
| `docs/pages/development/fern-website.md` | Full architecture guide |
