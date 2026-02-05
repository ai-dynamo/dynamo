---
orphan: true
---

# Documentation Templates

Templates for creating consistent Dynamo documentation.

## Directory Hierarchy

### Components (Router, Planner, KVBM, Frontend, Profiler)

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1: components/src/dynamo/<component>/README.md          │ ← Redirect stub
│   Content: 1-5 lines pointing to docs/components/<component>/│
│   Template: incode_readme.md                                 │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│ Tier 2: docs/components/<component>/                         │ ← User docs
│   • README.md ← component_readme.md                          │
│   • <component>_guide.md ← component_guide.md                │
│   • <component>_examples.md ← component_examples.md          │
└─────────────────────┬────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│ Tier 3: docs/design_docs/<component>_design.md               │ ← Contributor docs
│   Template: component_design.md                              │
└──────────────────────────────────────────────────────────────┘
```

### Backends (vLLM, SGLang, TRT-LLM)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: components/src/dynamo/<backend>/README.md   │ ← Redirect stub
│   Content: 1-5 lines pointing to docs/backends/     │
│   Template: incode_readme.md                        │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/backends/<backend>/                    │ ← User docs
│   • README.md ← backend_readme.md                   │
│   • <backend>_guide.md ← backend_guide.md           │
│                                                     │
│ Tier 2.5: docs/backends/README.md (exists)          │
│   • Backend comparison table                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 3: External                                    │
│   Backend internals documented in upstream repos    │
└─────────────────────────────────────────────────────┘
```

### Features (Multimodal, LoRA, Speculative Decoding)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: N/A                                         │
│   No in-code README (features are not components)   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/features/<feature>/                    │ ← User docs
│   • README.md ← feature_readme.md                   │
│   • <feature>_vllm.md ← feature_backend.md          │
│   • <feature>_sglang.md ← feature_backend.md        │
│   • <feature>_trtllm.md ← feature_backend.md        │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Tier 3: docs/design_docs/<feature>_design.md        │ ← Optional
│   Only if significant architecture                  │
└─────────────────────────────────────────────────────┘
```

### Integrations (LMCache, HiCache, NIXL)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: N/A                                         │
│   No in-code README (external tools)                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/integrations/<integration>/            │ ← User docs
│   • README.md ← integration_readme.md               │
│   • <integration>_setup.md (custom)                 │
│   • <integration>_<backend>.md (custom)             │
└─────────────────────────────────────────────────────┘
```

### Deploy (Kubernetes, Helm, Operator)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: N/A                                         │
│   No in-code README (deployment topics)             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/deploy/                                │ ← User docs
│   • README.md (deployment overview)                 │
│   • installation_guide.md, dynamo_operator.md       │
│   • helm.md, examples/                              │
└─────────────────────────────────────────────────────┘
```

### Performance (Tuning, Benchmarks)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: N/A                                         │
│   No in-code README (performance topics)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/performance/                           │ ← User docs
│   • README.md (performance overview)                │
│   • tuning.md, benchmarking.md, etc.                │
└─────────────────────────────────────────────────────┘
```

### Infrastructure (Observability, Fault Tolerance, Development)

```
┌─────────────────────────────────────────────────────┐
│ Tier 1: N/A                                         │
│   No in-code README (operations topics)             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Tier 2: docs/infrastructure/<topic>/                │ ← User docs
│   • README.md ← infrastructure_readme.md            │
│   • <subtopic>.md (detailed guides)                 │
└─────────────────────────────────────────────────────┘
```

## Three-Tier Pattern

| Tier | Purpose | Audience | Location |
|------|---------|----------|----------|
| **Tier 1** | Redirect stub (5 lines) | Developers browsing code | `components/src/dynamo/<name>/README.md` |
| **Tier 2** | User documentation | Users, operators | `docs/<category>/<name>/` (e.g., `docs/components/router/`) |
| **Tier 3** | Design documentation | Contributors | `docs/design_docs/<name>_design.md` |

## Template Selection

| What you're documenting | Templates to use |
|------------------------|------------------|
| New component | `incode_readme.md` + `component_*.md` (all 4) |
| New backend | `incode_readme.md` + `backend_*.md` (both) |
| New feature | `feature_readme.md` + `feature_backend.md` (per backend) |
| New integration | `integration_readme.md` |
| New deploy topic | Custom (follows `docs/deploy/` structure) |
| New performance topic | Custom (follows `docs/performance/` structure) |
| New infrastructure topic | `infrastructure_readme.md` |
| Migrating existing docs | Use the template matching your target file |

## Usage

1. Identify which category your documentation belongs to (component, backend, feature, integration)
2. Create the directory structure shown above
3. Copy templates to the correct locations with correct filenames
4. Replace all `<placeholders>` with actual values
5. Replace `<!-- comments -->` with actual content
6. Remove sections that don't apply

## Updating Navigation

After adding new documentation:

1. **Sphinx (current):** Update `docs/index.rst` or the appropriate `_sections/*.rst` file to include your new docs in the navigation
2. **Fern (future):** Update `fern/docs.yml` with your new pages

See [docs/README.md](../README.md) for documentation build instructions.

## Migrating Existing Docs

For migrating existing documentation to the new structure:

### Quick Reference

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Comprehensive migration guide with link update checklist
- [EXAMPLE_SKILL.md](EXAMPLE_SKILL.md) - AI-assisted migration skill (works with Cursor and Claude)

### Using the AI Skill

1. **Cursor IDE:** Copy `EXAMPLE_SKILL.md` to `.cursor/skills/docs-migration/SKILL.md`
2. **Claude/Other AI:** Copy the skill content into your conversation as context
3. Follow the phased approach with STOP points for review

### Manual Migration

1. Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for the full process
2. Use the link update checklist to update Sphinx and Fern navigation
3. Apply style guidelines for FLOW, STYLE, and CONSISTENCY
