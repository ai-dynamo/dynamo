---
name: rm-dynamo-docs
description: Remove a page from the Dynamo Fern docs site. Use when deleting documentation pages.
---

# Remove a Dynamo Docs Page

Claude Code skill for removing a page from the Dynamo Fern documentation site.

## Related Skills

| Skill | Use When |
|-------|----------|
| [add-dynamo-docs](../add-dynamo-docs/SKILL.md) | Adding a new docs page |
| [update-dynamo-docs](../update-dynamo-docs/SKILL.md) | Editing an existing docs page |

---

## Branch Rule

**ALL edits happen on `main` (or a feature branch based on `main`).**
The `docs-website` branch is CI-managed and must **never** be edited by hand.

## Working Directory

Must be in the `dynamo` repo (not `dynamo-tpm`). Architecture details: `docs/README.md`.

## When Invoked

### 1. Identify the Page

Ask for the page to remove (accepts any of):
- File path (e.g., `docs/pages/guides/old-page.md`)
- Page title (e.g., "Old Page")
- Topic keyword to search for

If given a title or keyword, search for the page:
```bash
# Search by title in navigation
grep -n "<title>" docs/versions/dev.yml

# Search by keyword in content
grep -rl "<keyword>" docs/pages/
```

### 2. Find the Navigation Entry

Locate the page's entry in `docs/versions/dev.yml`:

```bash
grep -n "<filename>" docs/versions/dev.yml
```

Note the exact `- page:` block and its indentation level. If the page is the
sole entry in a `- section:` block, the entire section should be removed.

### 3. Check for Incoming Links

Search for references to this page from other docs:

```bash
# Search for the filename across all docs pages
grep -r "<filename>" docs/pages/ --include="*.md"

# Also check the navigation file for any cross-references
grep -r "<filename>" docs/versions/
```

Report any files that link to the page being removed — these links will break
and need updating.

### 4. Remove the Markdown File

```bash
git rm docs/pages/<subdirectory>/<filename>.md
```

### 5. Remove the Navigation Entry

Edit `docs/versions/dev.yml` and delete the `- page:` block (and its `path:`
line). If this was the last page in a section, remove the entire `- section:`
block.

### 6. Fix Broken Incoming Links

For each file that linked to the removed page:
- Remove the link, or
- Redirect to a replacement page, or
- Leave a note about the removal

### 7. Validate

```bash
cd docs
ln -sf . fern  # symlink required by Fern CLI
fern check
fern docs broken-links
```

### 8. Preview Locally (Optional)

```bash
cd docs
fern docs dev
```

Opens a local preview at `http://localhost:3000` with hot reload. No token required.

### 9. Commit

```bash
git add -u docs/
git commit -s -m "docs: remove <page-title> page"
```

## Debugging

### `fern check` fails

- **Orphaned nav entry:** You deleted the file but left the `- page:` entry in `dev.yml`. Remove it.
- **Empty section:** If the removed page was the only entry in a section, delete the entire `- section:` block from `dev.yml`.

### `fern docs broken-links` reports errors

- **Incoming links to removed page:** Other pages still link to the deleted file. Search with `grep -r "<filename>" docs/pages/` and update or remove those links.

### CI fails after merge

- **MDX parse error:** Angle-bracket URLs like `<https://example.com>` break MDX parsing. Use `[text](https://example.com)` instead.
- **Broken links check:** The `detect_broken_links.py` job found pages that still reference the removed file. Fix all incoming links before merging.
- **Fern publish error:** Check the Actions tab for the `Fern Docs` workflow. Common causes: expired `FERN_TOKEN`, invalid `docs.yml` syntax.

### Page still appears on the live site

- **Sync delay:** After merge to `main`, the sync-dev workflow takes a few minutes to publish.
- **Cached version:** Fern CDN may cache the old page briefly. Hard-refresh or wait a few minutes.

## Key References

| File | Purpose |
|------|---------|
| `docs/versions/dev.yml` | Navigation tree — remove entries here |
| `docs/pages/` | Content directory — delete pages here |
| `docs/README.md` | Full architecture guide |
