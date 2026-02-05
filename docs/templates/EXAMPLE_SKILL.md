---
orphan: true
---

# Documentation Migration Skill

This file is a ready-to-use Cursor skill for AI-assisted documentation migration.

---

## How to Use This Skill

### Option 1: Cursor IDE

1. Create the skill directory:
   ```bash
   mkdir -p .cursor/skills/docs-migration
   ```

2. Copy this file:
   ```bash
   cp docs/templates/EXAMPLE_SKILL.md .cursor/skills/docs-migration/SKILL.md
   ```

3. Remove the `orphan: true` header and this "How to Use" section

4. The skill will be available when working on documentation migration

### Option 2: Claude (or any AI)

1. Copy everything below the separator line (`---`)
2. Paste into your conversation as context
3. Ask the AI: "Help me migrate the [component] documentation to the new structure"

---

## Skill Content

Copy everything below this line for use as an AI prompt:

---
name: docs-migration
description: Migrate Dynamo documentation to the 9-category hierarchy. Use when migrating components, backends, features, or other docs to the new structure.
---

# Documentation Migration

This skill guides you through migrating Dynamo documentation to the new 9-category hierarchy.

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| Component/Topic | Yes | What to migrate (e.g., "planner", "kubernetes", "multimodal") |
| Source Path | Yes | Current location (e.g., `docs/planner/`) |
| Target Category | Yes | One of: components, backends, features, deploy, performance, infrastructure, integrations |

## Directory Hierarchy

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

## Phase 1: Analyze Existing Docs

### Step 1.1: Inventory Current Files

```bash
# List existing documentation
ls -la docs/<source_path>/

# Count lines in each file
wc -l docs/<source_path>/*.md
```

### Step 1.2: Categorize Content

For each file, identify content type:

| Category | Target File | Description |
|----------|-------------|-------------|
| Overview | README.md | Component description, feature matrix |
| Quick Start | README.md | Minimal steps to get running |
| Deployment | `<name>_guide.md` | Setup, prerequisites, container images |
| Configuration | `<name>_guide.md` | CLI args, env vars, config files |
| Integration | `<name>_guide.md` | Connecting to other components |
| Troubleshooting | `<name>_guide.md` | Common issues and fixes |
| Examples | `<name>_examples.md` | Code samples, YAML configs |
| Architecture | `<name>_design.md` | Design decisions, algorithms |

**>>> STOP: Share your analysis. Ask if there are content priorities or known issues.**

---

## Phase 2: Create Migration Mapping

### Step 2.1: Document Current → Target Mapping

Create a mapping showing where each section will move:

```markdown
## Content Migration Mapping

### README.md (Tier 2)

| New Section | Source | Est. Lines |
|-------------|--------|------------|
| Overview | source_file.md → Section | X |
| Feature Matrix | source_file.md → Section | X |
| Quick Start | source_file.md → Section | X |
| Next Steps | New | 10 |

### <name>_guide.md (Tier 2)

| New Section | Source | Est. Lines |
|-------------|--------|------------|
| Deployment | source_file.md → Section | X |
| Configuration | source_file.md → Section | X |
| Integration | source_file.md → Section | X |
| Troubleshooting | source_file.md → Section | X |
```

**>>> STOP: Share mapping. Ask if any content should be prioritized or excluded.**

---

## Phase 3: Create File Structure

### Step 3.1: Create Target Directory

```bash
# For components
mkdir -p docs/components/<name>

# For other categories
mkdir -p docs/<category>/<name>
```

### Step 3.2: Create Files

```bash
touch docs/<category>/<name>/README.md
touch docs/<category>/<name>/<name>_guide.md
touch docs/<category>/<name>/<name>_examples.md
touch docs/design_docs/<name>_design.md
```

### Step 3.3: Create Tier 1 Stub (Components Only)

For components, create redirect stub:

```markdown
# Dynamo <Component>

<One-sentence description.>

See `docs/components/<name>/` for documentation.
```

---

## Phase 4: Migrate Content

### Step 4.1: Use Templates

Reference templates in `docs/templates/`:
- `component_readme.md` - Tier 2 README
- `component_guide.md` - Tier 2 Guide
- `component_examples.md` - Tier 2 Examples
- `component_design.md` - Tier 3 Design

### Step 4.2: Preserve All Content

- Copy content exactly unless errors exist
- Preserve code examples
- Preserve diagrams (Mermaid, images)
- Update internal links to new paths

**>>> STOP: Share migrated documents. Ask if content is complete.**

---

## Phase 5: Update Links

### Step 5.1: Find Files Linking to Old Path

```bash
# Find all files with links to old path
rg -l "docs/<old_path>" docs/ fern/

# Find RST cross-references
rg ":doc:\`.*<old_path>" docs/

# Find relative markdown links
rg "\]\(.*<old_path>" docs/
```

### Step 5.2: Update Sphinx Navigation

1. **index.rst** - Update toctree entries:
   ```rst
   .. toctree::
      Page Title <../new/path/file>
   ```

2. **_sections/*.rst** - Update section toctrees

3. **conf.py** - Add redirect for moved files:
   ```python
   redirects = {
       "old/path/file": "../new/path/file.html",
   }
   ```

### Step 5.3: Update Fern Navigation

1. **versions/next.yml** - Update page paths:
   ```yaml
   - page: Page Title
     path: ../pages/new/path.md
   ```

2. **Move files** in `fern/pages/` to match new structure

### Step 5.4: Update Cross-References in Other Docs

For each file found in Step 5.1:
- Update relative paths to new locations
- Verify links work

**>>> STOP: Share link update summary. List files modified.**

---

## Phase 6: Edit for Style

Review migrated documents for FLOW, STYLE, and CONSISTENCY.

**Do NOT change content meaning - only improve presentation.**

### Step 6.1: Review Checklist

For each document:

**FLOW:**
- [ ] First paragraph states what the doc covers
- [ ] Sections ordered: Overview → Setup → Usage → Troubleshooting
- [ ] No orphaned paragraphs (single sentences between sections)

**STYLE:**
- [ ] Instructions use active voice ("Run", "Create", "Add")
- [ ] No redundant phrases ("To" not "In order to")
- [ ] Sentences ≤25 words

**CONSISTENCY:**
- [ ] Component names: vLLM, SGLang, TensorRT-LLM
- [ ] Status indicators: ✅ 🚧 ❌
- [ ] Heading hierarchy: # → ## → ### (no skips)
- [ ] Code blocks specify language

### Step 6.2: Generate Suggested Edits

Present suggestions using FLAG format:

```markdown
---

### FLAG: [FLOW|STYLE|CONSISTENCY] - [Brief Description]

**File:** `path/to/file.md`
**Line(s):** X-Y

**Current:**
> [Original text]

**Suggested:**
> [Improved text]

**Reasoning:** [Why this improves flow/style/consistency]

---
```

### Step 6.3: Apply Approved Edits

After user reviews:
- Apply approved edits only
- Skip rejected suggestions
- Document patterns for future reference

**>>> STOP: Share suggested edits. Ask which to apply.**

---

## Phase 7: Validate and Cleanup

### Step 7.1: Validation Checklist

```
Validation for: [COMPONENT_NAME]
- [ ] All content from original files preserved
- [ ] No broken links (test with docs build)
- [ ] Feature matrix matches current capabilities
- [ ] Code examples are correct
- [ ] Mermaid diagrams render
- [ ] Navigation links work between files
- [ ] Sphinx toctree updated
- [ ] Fern navigation updated
- [ ] conf.py redirects added
```

### Step 7.2: Test Docs Build

```bash
# Build Sphinx docs
cd docs && make html

# Check for warnings about missing references
```

### Step 7.3: Cleanup Old Files

After validation and approval:
1. Delete original files
2. Keep deprecated files with deprecation notice if needed
3. Commit changes

**>>> STOP: Share validation results. Ask for approval before deleting originals.**

---

## Category-Specific Notes

### Components (Router, Planner, KVBM, Frontend, Profiler)

- Target: `docs/components/<name>/`
- Requires Tier 1 stub in `components/src/dynamo/<name>/README.md`
- Tier 3 design doc in `docs/design_docs/<name>_design.md`

### Backends (vLLM, SGLang, TRT-LLM)

- Target: `docs/backends/<name>/`
- Tier 3 is external (upstream project docs)
- Create `docs/backends/README.md` for backend comparison

### Deploy (Kubernetes)

- Target: `docs/deploy/`
- Flat structure (no subdirectories per topic)
- Examples go in `docs/deploy/examples/`

### Performance

- Target: `docs/performance/`
- Includes tuning and benchmarks (merged)
- Flat structure

### Infrastructure (Observability, Fault Tolerance, Development)

- Target: `docs/infrastructure/<topic>/`
- Subdirectory per topic
- Development guides for contributors
