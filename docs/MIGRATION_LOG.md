# Sphinx to Fern Migration - Execution Log

**Migration Date:** 2026-02-04
**Branch:** `fern-folder-migration`
**Executed by:** Claude

---

## Summary

Successfully migrated documentation from Sphinx (`docs/`) to Fern. The `fern/` folder has been moved to `docs/`, replacing the Sphinx-based documentation structure.

---

## Phase 1: Pre-Migration - Add Missing Content ✅

**Completed:** 2026-02-04

### Problem Identified

Two documentation files existed in Sphinx `docs/` but were missing from `fern/`:

| File | Created | PR | Root Cause |
|------|---------|-----|------------|
| `backends/sglang/diffusion-lm.md` | 2026-01-21 | #5533 | Added before fern migration but missed |
| `router/kv_events.md` | 2026-01-13 | #5386 | Added before fern migration but missed |

### Timeline Context

```
2026-01-13  router/kv_events.md created (PR #5386)
2026-01-21  backends/sglang/diffusion-lm.md created (PR #5533)
2026-01-26  fern/ initial migration (PR #5445) <-- both files missed
2026-01-28  diffusion-lm.md updated (PR #5723)
2026-02-03  Latest fern updates
2026-02-04  This migration executed
```

### Actions Taken

1. **Copied missing files to fern:**
   ```bash
   cp docs/backends/sglang/diffusion-lm.md fern/pages/backends/sglang/
   cp docs/router/kv_events.md fern/pages/router/kv-events.md
   ```

2. **Updated navigation** in `fern/versions/next.yml`:
   - Added "Diffusion LM" page under `Backend Details > SGLang` section
   - Added "KV Events" page under `Router Details` section

3. **Verified content parity** - All docs content now has fern equivalents:
   - Files with different locations (relocated, not missing):
     - `design_docs/request_plane.md` → `guides/request-plane.md`
     - `development/jail_stream.md` → `guides/jail-stream-readme.md`
     - `kubernetes/deployment/minikube.md` → `kubernetes/deployment/minikube-setup.md`
   - Meta files not needed in fern: `README.md`, `SPHINX_TO_FERN_MIGRATION.md`

---

## Phase 2: Create Backup ✅

**Completed:** 2026-02-04

### Backups Created

| Backup Type | Location | Purpose |
|-------------|----------|---------|
| **Git Tag** | `sphinx-docs-backup-20260204` | Restore with `git checkout sphinx-docs-backup-20260204 -- docs/` |
| **Local Folder** | `docs.sphinx.backup/` | Full copy of original docs (44,949 files) |

### Rollback Commands (if needed)

```bash
# Option 1: Restore from local backup
rm -rf docs
mv docs.sphinx.backup docs

# Option 2: Restore from git tag
git checkout sphinx-docs-backup-20260204 -- docs/
```

---

## Phase 3: Perform Migration ✅

**Completed:** 2026-02-04

### Actions Taken

```bash
# Remove old Sphinx docs
rm -rf docs

# Move fern to docs
mv fern docs
```

### New docs/ Structure

```
docs/
├── .gitignore
├── docs.yml              # Fern configuration (title, colors, logo)
├── fern.config.json      # Fern version config
├── assets/
│   └── img/              # 36 images (PNG/SVG)
├── pages/                # 113 markdown files
│   ├── agents/
│   ├── api/
│   ├── backends/
│   ├── benchmarks/
│   ├── design-docs/
│   ├── development/
│   ├── fault-tolerance/
│   ├── frontends/
│   ├── getting-started/
│   ├── guides/
│   ├── kubernetes/
│   ├── kvbm/
│   ├── mocker/
│   ├── multimodal/
│   ├── observability/
│   ├── performance/
│   ├── planner/
│   ├── reference/
│   └── router/
└── versions/
    └── next.yml          # Navigation structure
```

### Verification Results

| Check | Result |
|-------|--------|
| Config files present | ✅ docs.yml, fern.config.json, versions/next.yml |
| Missing files added | ✅ diffusion-lm.md, kv-events.md present |
| Total markdown files | 113 |
| Total images | 36 |
| Git status | 339 files changed (deletions + renames) |

---

## What Changed

### Removed (Sphinx-specific)

- `docs/conf.py` - Sphinx configuration
- `docs/generate_docs.py` - Sphinx build script
- `docs/Makefile` - Sphinx make targets
- `docs/_extensions/` - Custom Sphinx extensions
- `docs/_sections/` - Sphinx toctree files (*.rst)
- `docs/_includes/` - Sphinx include directives (*.rst)
- `docs/_static/` - Sphinx static files
- `docs/*.rst` - All RST files
- `docs/node_modules/` - Docusaurus artifacts
- `docs/build/` - Build output
- Various config files (exclusions.txt, repositories.txt, etc.)

### Added (Fern-specific)

- `docs/docs.yml` - Fern docs configuration
- `docs/fern.config.json` - Fern version/org config
- `docs/versions/next.yml` - Navigation structure
- `docs/pages/` - All documentation in kebab-case structure
- `docs/assets/img/` - Images with kebab-case names

### Naming Convention Changes

| Old (Sphinx) | New (Fern) |
|--------------|------------|
| `design_docs/` | `design-docs/` |
| `fault_tolerance/` | `fault-tolerance/` |
| `api/nixl_connect/` | `api/nixl-connect/` |
| `kv_cache_routing.md` | `kv-cache-routing.md` |
| `dynamo_operator.md` | `dynamo-operator.md` |

---

## Phase 4: Handle Special Files ✅

**Completed:** 2026-02-04

### Files Evaluated

| File | Location in Backup | Decision | Action |
|------|-------------------|----------|--------|
| `openapi.json` | `frontends/openapi.json` | **Keep** - Generated API spec referenced in README | Copied to `docs/frontends/openapi.json` |
| `package-lock.json` | Root | **Remove** - Docusaurus artifact | Not copied |
| `switcher.json` | `_static/` | **Remove** - Sphinx theme config | Not copied |
| `project.json` | Root | **Remove** - Sphinx build config | Not copied |
| `examples/` | Symlinks to repo examples | **Remove** - Fern uses GitHub links | Not copied |

### Actions Taken

1. **Copied `openapi.json`:**
   ```bash
   mkdir -p docs/frontends
   cp docs.sphinx.backup/frontends/openapi.json docs/frontends/
   ```

   This file is generated by:
   ```bash
   cargo run -p dynamo-llm --bin generate-frontend-openapi
   ```
   And is referenced in the root README.md.

2. **Verified `.gitignore`** - Appropriate for Fern:
   ```
   **/.preview
   **/.definition
   !*.svg
   ```

3. **Empty `.github/workflows/`** - No action needed (fern-specific workflows can be added later)

### Updated docs/ Structure

```
docs/
├── .gitignore
├── docs.yml
├── fern.config.json
├── frontends/
│   └── openapi.json      # ← Added (generated API spec)
├── assets/
│   └── img/
├── pages/
│   └── ...
└── versions/
    └── next.yml
```

---

## Phase 5: Update Repository References ✅

**Completed:** 2026-02-04

### README.md Updates

Updated all documentation links from old Sphinx paths to new Fern paths:

| Old Path | New Path |
|----------|----------|
| `docs/images/` | `docs/assets/img/` |
| `docs/design_docs/` | `docs/pages/design-docs/` |
| `docs/reference/` | `docs/pages/reference/` |
| `docs/kubernetes/` | `docs/pages/kubernetes/` |
| `docs/benchmarks/` | `docs/pages/benchmarks/` |
| `docs/planner/` | `docs/pages/planner/` |
| `docs/fault_tolerance/` | `docs/pages/fault-tolerance/` |
| `docs/router/` | `docs/pages/router/` |
| `docs/kvbm/` | `docs/pages/kvbm/` |
| `docs/agents/` | `docs/pages/agents/` |

### Other Files Updated

| File | Change |
|------|--------|
| `.github/filters.yaml` | Updated `docs/kubernetes/api_reference.md` → `docs/pages/kubernetes/api-reference.md` |
| `deploy/operator/config/crd/bases/*.yaml` | Updated autoscaling doc references |
| `deploy/helm/charts/crds/templates/*.yaml` | Updated autoscaling doc references |
| `examples/backends/*/deploy/disagg_planner.yaml` | Updated planner doc references |
| `components/src/dynamo/planner/utils/perf_interpolation.py` | Updated profiling doc path |
| `benchmarks/profiler/webui/utils.py` | Updated planner doc GitHub URL |

### Fern CLI Compatibility

Created symlink `fern -> docs` at repository root to enable Fern CLI:

```bash
ln -sf docs fern
```

The Fern CLI expects a `fern/` directory. This symlink allows both:
- `docs/` folder for conventional documentation location
- `fern` symlink for Fern CLI compatibility

---

## Phase 6: Verification ✅

**Completed:** 2026-02-04

### Fern Check Results

```bash
$ fern check --warnings
[docs] 1 warning
    [warning] The contrast ratio between the accent color and the background 
              color for light mode is 5.61:1. Fern will adjust the color to 
              meet the minimum contrast ratio of 7:1 for WCAG AAA.

Found 0 errors and 1 warning in 0.001 seconds.
```

**Result:** ✅ 0 errors, 1 minor accessibility warning (color contrast auto-adjusted by Fern)

### Structure Verification

| Check | Result |
|-------|--------|
| `fern check` | ✅ 0 errors |
| docs.yml present | ✅ |
| fern.config.json present | ✅ |
| versions/next.yml present | ✅ |
| pages/ folder with 113+ md files | ✅ |
| assets/img/ folder with 36 images | ✅ |
| fern symlink works | ✅ |

---

## Remaining Tasks

### Phase 7: Cleanup and Commit (TODO)

- [ ] Remove `docs.sphinx.backup/` after PR merge
- [ ] Commit changes
- [ ] Create PR

### CI/CD Notes (Separate PR Recommended)

The following files still reference Sphinx and should be updated in a separate PR:
- `.github/workflows/generate-docs.yml` - Uses Sphinx build commands
- `container/Dockerfile.docs` - Uses Sphinx dependencies
- `pyproject.toml` - May have Sphinx dependencies in docs extras

---

## Files Reference

| File | Description |
|------|-------------|
| `docs/MIGRATION_LOG.md` | This execution log |
| `docs.sphinx.backup/SPHINX_TO_FERN_MIGRATION.md` | Original migration plan |
| `docs.sphinx.backup/` | Backup of original Sphinx docs |
| `fern` | Symlink to `docs/` for Fern CLI compatibility |

---

## Quick Reference: Fern Commands

```bash
# Preview docs locally (from repo root)
fern docs dev

# Check for errors (from repo root)
fern check

# Check with warnings
fern check --warnings
```
