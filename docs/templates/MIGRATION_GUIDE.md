---
orphan: true
---

# Documentation Migration Guide

This guide covers migrating Dynamo documentation to the new 9-category hierarchy.

---

## Directory Hierarchy

Documentation is organized into 9 top-level categories:

```
docs/
├── components/          # Router, Planner, KVBM, Frontend, Profiler
├── backends/            # vLLM, SGLang, TRT-LLM
├── features/            # Multimodal, LoRA, Speculative Decoding
├── deploy/              # Kubernetes, Helm, Operator
├── performance/         # Tuning, Benchmarks
├── infrastructure/      # Observability, Fault Tolerance, Development
├── integrations/        # LMCache, HiCache, NIXL
├── reference/           # CLI, Glossary, Support Matrix
└── design_docs/         # Tier 3 design documents
```

---

## Category Reference

| Category | Location | Content Type |
|----------|----------|--------------|
| **Components** | `docs/components/<name>/` | Standalone deployable services (Router, Planner, KVBM, Frontend, Profiler) |
| **Backends** | `docs/backends/<name>/` | LLM inference engine integrations (vLLM, SGLang, TRT-LLM) |
| **Features** | `docs/features/<name>/` | Cross-cutting capabilities (Multimodal, LoRA, Speculative Decoding) |
| **Deploy** | `docs/deploy/` | Kubernetes deployment, operator, Helm charts |
| **Performance** | `docs/performance/` | Performance tuning, benchmarking, profiling |
| **Infrastructure** | `docs/infrastructure/<topic>/` | Observability, fault tolerance, development guides |
| **Integrations** | `docs/integrations/<name>/` | External tool integrations (LMCache, HiCache, NIXL) |
| **Reference** | `docs/reference/` | CLI reference, glossary, support matrix |
| **Design** | `docs/design_docs/` | Architecture and algorithm documentation (Tier 3) |

---

## Three-Tier Pattern

Components and backends follow a three-tier documentation pattern:

| Tier | Location | Purpose | Audience |
|------|----------|---------|----------|
| **Tier 1** | `components/src/dynamo/<name>/README.md` | Redirect stub (5 lines) | Developers browsing code |
| **Tier 2** | `docs/<category>/<name>/` | User documentation | Users, operators |
| **Tier 3** | `docs/design_docs/<name>_design.md` | Design documentation | Contributors |

---

## Link Update Checklist

When moving documentation, update links in these locations:

### 1. Internal Markdown Links

Find files linking to the old path:

```bash
# Find all files with links to old path
rg -l "docs/old_path" docs/ fern/

# Find relative markdown links
rg "\]\(.*old_path" docs/
```

### 2. Sphinx Configuration

**Files to update:**

| File | Content |
|------|---------|
| `docs/index.rst` | Main toctree with section references |
| `docs/_sections/*.rst` | Section toctrees (8 files) |
| `docs/hidden_toctree.rst` | Orphaned pages not in main nav |
| `docs/conf.py` | Redirects mapping (lines 40-98) |

**Toctree syntax:**
```rst
.. toctree::
   :hidden:

   Page Title <../new/path/file>
```

**Add redirect in conf.py:**
```python
redirects = {
    "old/path/file": "../new/path/file.html",
}
```

### 3. Fern Configuration

**Files to update:**

| File | Content |
|------|---------|
| `fern/docs.yml` | Site config and version reference |
| `fern/versions/next.yml` | Full navigation structure |

**Navigation syntax:**
```yaml
- page: Page Title
  path: ../pages/new/path.md
```

**Move files in `fern/pages/` to match new structure.**

### 4. RST Cross-References

Find and update:
```rst
:doc:`../old/path/file`
```

### 5. Include Directives

Check `docs/_includes/` for includes:
```rst
.. include:: ../old/path/file.rst
```

---

## Pre-Migration Link Validation

Before migrating, validate source docs to avoid carrying over broken links.

### Pre-flight Broken Link Check

```bash
# Install lychee (if not available)
cargo install lychee   # or: brew install lychee

# Check source files (example: migrating kvbm docs)
lychee docs/kvbm/ --offline --exclude-path docs/_build

# Or use the full check with external URLs
lychee docs/kvbm/ --exclude-path docs/_build
```

If lychee is unavailable, use ripgrep to find potentially broken links:

```bash
# Find all internal markdown links and spot-check targets
rg -n '\]\([^http][^)]*\.md' docs/kvbm/
```

### Golden Rule

**Only link to files that exist.** Before adding any link:

1. Verify the target file exists at the expected path
2. Test the relative path calculation (count `../` correctly)
3. For cross-section links, consider using the cross-reference path table

### Post-Migration Validation

After moving files, run link check again to catch broken references:

```bash
# Check all docs after migration
lychee docs/ --offline --exclude-path docs/_build

# Check specific migrated directory (example: after moving to components/kvbm)
lychee docs/components/kvbm/ --offline
```

---

## Style Editing Guidelines

After migrating content, review for FLOW, STYLE, and CONSISTENCY.

**Do NOT change content meaning - only improve presentation.**

### FLOW Rules

| Rule | Description |
|------|-------------|
| Lead with the point (BLUF) | First paragraph states what the doc covers |
| Logical section order | Overview → Setup → Usage → Troubleshooting |
| One idea per paragraph | Split paragraphs with multiple topics |
| No orphaned sentences | Avoid single sentences between sections |

### STYLE Rules

| Rule | Example |
|------|---------|
| Active voice for instructions | "Run the command" not "The command should be run" |
| Consistent tense | All steps in present tense |
| No redundant phrases | "To" not "In order to" |
| Short sentences | Target ≤25 words |

### CONSISTENCY Rules

| Rule | Standard |
|------|----------|
| Component names | vLLM, SGLang, TensorRT-LLM (or TRT-LLM) |
| Status indicators | ✅ Supported, 🚧 Experimental, ❌ Not Supported |
| Heading hierarchy | # → ## → ### (no skips) |
| Code block languages | Always specify (```python, ```bash, ```yaml) |

---

## Related Files

- [SOURCE_TARGET_MAPPING.md](SOURCE_TARGET_MAPPING.md) - Comprehensive file-level source → target mapping
- [README.md](README.md) - Template overview and selection guide
- [EXAMPLE_SKILL.md](EXAMPLE_SKILL.md) - Cursor skill for AI-assisted migration
- Individual templates: `component_readme.md`, `component_guide.md`, etc.
