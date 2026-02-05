# Sphinx to Fern Documentation Migration Plan

## Executive Summary

This document outlines the plan to migrate documentation from Sphinx (`docs/`) to Fern (`fern/`). The Fern docs were initially migrated on **2026-01-26** (PR #5445) and have been receiving updates since. The migration involves replacing `docs/` contents with `fern/` contents.

---

## 1. Content Gap Analysis

### Files in `docs/` with NO equivalent in `fern/`

| File | Created | PR | Status | Action |
|------|---------|-----|--------|--------|
| `backends/sglang/diffusion-lm.md` | 2026-01-21 | #5533 (feat: v0 diffusion handler support) | **Added AFTER fern migration** | Must copy to fern |
| `router/kv_events.md` | 2026-01-13 | #5386 (docs: kv events) | **Added BEFORE fern migration but missed** | Must copy to fern |

### Timeline Context

```
2026-01-13  router/kv_events.md created (PR #5386)
2026-01-21  backends/sglang/diffusion-lm.md created (PR #5533)
2026-01-26  fern/ initial migration (PR #5445) <-- kv_events existed but was missed
2026-01-28  diffusion-lm.md updated (PR #5723)
2026-02-03  Latest fern updates (ChReK, SGLang links, etc.)
```

### Files with Different Locations (already handled)

| docs/ Location | fern/ Location | Status |
|----------------|----------------|--------|
| `development/jail_stream.md` | `guides/jail-stream-readme.md` | OK - relocated |
| `design_docs/request_plane.md` | `guides/request-plane.md` | OK - relocated |

### Files in `fern/` that are NEW (not in docs/)

| File | Notes |
|------|-------|
| `getting-started/quickstart.md` | New consolidated quickstart |
| `getting-started/installation.md` | New installation guide |
| `getting-started/examples.md` | New examples page |
| `kubernetes/quickstart.md` | New K8s quickstart |
| `kvbm/kvbm-intro.md` | New intro (replaces `.rst`) |
| `planner/planner-intro.md` | New intro (replaces `.rst`) |

---

## 2. Naming Convention Changes

Fern uses **kebab-case** (hyphens) instead of **snake_case** (underscores):

| docs/ (Sphinx) | fern/ |
|----------------|-------|
| `design_docs/` | `design-docs/` |
| `fault_tolerance/` | `fault-tolerance/` |
| `api/nixl_connect/` | `api/nixl-connect/` |
| `kv_cache_routing.md` | `kv-cache-routing.md` |
| `dynamo_operator.md` | `dynamo-operator.md` |

---

## 3. Files Comparison Summary

| Category | docs/ (Sphinx) | fern/ | Notes |
|----------|----------------|-------|-------|
| Markdown pages | ~95 | ~111 | Fern has more (reorganized) |
| Images | 29 | 36 | Fern has additional assets |
| Config files | conf.py, Makefile, etc. | docs.yml, fern.config.json, next.yml | Different systems |
| RST files | 14 | 0 | Converted to MD in fern |

---

## 4. Migration Steps

### Phase 1: Pre-Migration - Add Missing Content to Fern

**1.1 Copy missing files to fern:**

```bash
# Copy diffusion-lm.md (added after fern migration)
cp docs/backends/sglang/diffusion-lm.md fern/pages/backends/sglang/

# Copy kv_events.md (missed during fern migration)
cp docs/router/kv_events.md fern/pages/router/kv-events.md
```

**1.2 Update `fern/versions/next.yml` to add missing pages:**

Add to SGLang backend section (around line 302):
```yaml
              - page: Diffusion LM
                path: ../pages/backends/sglang/diffusion-lm.md
```

Add to Router Details section (around line 226):
```yaml
              - page: KV Events
                path: ../pages/router/kv-events.md
```

**1.3 Verify content parity:**
```bash
# Quick diff to ensure no other content was missed
diff -rq docs/ fern/pages/ --exclude="*.rst" --exclude="*.py" --exclude="node_modules" --exclude="build" --exclude="_*"
```

### Phase 2: Create Backup

```bash
# Create a backup branch/tag
git checkout -b sphinx-to-fern-migration
git tag sphinx-docs-backup

# Or create local backup
cp -r docs docs.sphinx.backup
```

### Phase 3: Perform Migration

**Option A: Replace docs/ with fern/ contents**
```bash
# Remove Sphinx-specific files and content
rm -rf docs/_extensions docs/_includes docs/_sections docs/_static
rm -rf docs/node_modules docs/build docs/src
rm -f docs/conf.py docs/generate_docs.py docs/Makefile
rm -f docs/*.rst docs/**/*.rst
rm -f docs/exclusions.txt docs/repositories.txt docs/project.json
rm -f docs/package-lock.json docs/package.json

# Copy fern structure into docs
cp fern/docs.yml docs/
cp fern/fern.config.json docs/
cp -r fern/versions docs/
cp -r fern/assets docs/
cp -r fern/pages docs/

# Remove old markdown files (they're now in pages/)
# ... careful cleanup of old structure
```

**Option B: Replace entire directory (Recommended)**
```bash
# Rename directories
mv docs docs.sphinx.old
mv fern docs

# Verify structure
ls docs/
# Should show: assets/ docs.yml fern.config.json pages/ versions/
```

### Phase 4: Handle Special Files

**4.1 Check if openapi.json is needed:**
```bash
# This file exists in Sphinx docs - verify if Fern needs it
ls -la docs.sphinx.old/frontends/openapi.json
# If needed, copy to appropriate fern location
```

**4.2 Update .gitignore if needed:**
```bash
# Fern has its own .gitignore
cat docs/.gitignore
```

### Phase 5: Update Repository References

**5.1 Update root README.md** (if it references docs build):
- Change Sphinx build commands to Fern commands
- Update documentation URLs

**5.2 Update CI/CD pipelines:**
- Search for Sphinx references in `.github/workflows/`
- Replace with Fern build commands:
  ```yaml
  # Old (Sphinx)
  - run: cd docs && make html
  
  # New (Fern)
  - run: cd docs && fern generate --docs
  ```

**5.3 Update pyproject.toml or package.json:**
- Remove Sphinx dependencies if present
- Add Fern CLI if needed

### Phase 6: Verification

```bash
# Build Fern docs
cd docs
fern generate --docs

# Or use Fern's preview
fern docs dev
```

**Verify:**
- [ ] All pages render correctly
- [ ] Images load properly
- [ ] Navigation works as expected
- [ ] No broken links
- [ ] diffusion-lm.md is accessible
- [ ] kv-events.md is accessible

### Phase 7: Cleanup

```bash
# After successful verification
rm -rf docs.sphinx.old
# or
rm -rf docs.sphinx.backup

# Commit changes
git add -A
git commit -m "docs: migrate from Sphinx to Fern

- Replace Sphinx documentation structure with Fern
- Add missing files: diffusion-lm.md, kv-events.md
- Update navigation in versions/next.yml

Closes #XXXX"
```

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| URL changes (snake_case → kebab-case) | Set up redirects if docs are published |
| CI/CD pipeline failures | Update workflows before merging |
| Missing content discovered late | Thorough diff comparison in Phase 1 |
| Build failures | Test Fern build locally before committing |

---

## 6. Rollback Plan

If migration fails:
```bash
# Restore from backup
rm -rf docs
mv docs.sphinx.backup docs
# or
git checkout sphinx-docs-backup -- docs/
```

---

## 7. Post-Migration Tasks

- [ ] Update any external links pointing to old doc structure
- [ ] Notify team of new documentation system
- [ ] Update CONTRIBUTING.md with new docs contribution workflow
- [ ] Remove Sphinx from any dependency files
- [ ] Update docs hosting configuration (if applicable)

---

## Appendix: Sphinx Files to Remove

These files are Sphinx-specific and should NOT be migrated:

```
docs/conf.py                    # Sphinx configuration
docs/generate_docs.py           # Sphinx build script
docs/Makefile                   # Sphinx make targets
docs/_extensions/               # Custom Sphinx extensions
docs/_sections/                 # Sphinx toctree files
docs/_includes/                 # Sphinx include directives
docs/_static/                   # Sphinx static files
docs/*.rst                      # All RST files
docs/**/*.rst                   # Nested RST files
docs/node_modules/              # Docusaurus artifacts
docs/build/                     # Build output
docs/exclusions.txt             # Sphinx-specific
docs/repositories.txt           # Sphinx-specific
docs/project.json               # Build config
docs/package-lock.json          # Node dependencies
docs/src/                       # Docusaurus source
docs/static/                    # Docusaurus static
```
