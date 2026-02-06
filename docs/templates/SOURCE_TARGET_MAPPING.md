---
orphan: true
---

# Source-to-Target File Mapping

This document provides a comprehensive file-level mapping from current documentation locations to the new hierarchy for Components, Backends, Features, and Integrations.

---

## How to Use This Mapping

1. Find the source file you want to migrate in the tables below
2. Note the **Target** path and **Action** type
3. Follow the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for link updates
4. Use [EXAMPLE_SKILL.md](EXAMPLE_SKILL.md) for AI-assisted migration

### Legend

| Symbol | Content Type |
|--------|--------------|
| **O** | Overview - entry point, introduction |
| **G** | Guide - step-by-step instructions |
| **E** | Examples - code samples, templates |
| **D** | Design - architecture, algorithms |
| **R** | Reference - API specs, CLI docs |

| Action | Description |
|--------|-------------|
| **Move** | Relocate file to new path |
| **Merge** | Combine multiple files into one |
| **Split** | Separate one file into multiple |
| **Convert** | Transform RST to Markdown |
| **Create** | New content needed |
| **Extract** | Pull content from another file |

---

## 1. Components

### Router (1,334 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/router/README.md` | 316 | O | `docs/components/router/README.md` | Move | Quick start, configuration |
| `docs/router/kv_cache_routing.md` | 733 | G | `docs/components/router/router_guide.md` | Move | Deep technical guide |
| `docs/router/kv_events.md` | 285 | G | `docs/components/router/router_guide.md` | Merge | Append to guide |

**Tier 1 (In-Code):**

| Source | Target | Action |
|--------|--------|--------|
| `components/src/dynamo/router/README.md` | Keep | Update link to `docs/components/router/` |

### Planner (863 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/planner/planner_intro.rst` | 82 | O | `docs/components/planner/README.md` | Convert | RST→MD, merge overview |
| `docs/planner/sla_planner_quickstart.md` | 521 | G | `docs/components/planner/planner_guide.md` | Split | Guide + examples |
| `docs/planner/sla_planner.md` | 203 | D | `docs/design_docs/planner_design.md` | Move | Architecture content |
| `docs/planner/load_planner.md` | 57 | G | `docs/components/planner/load_planner.md` | Move | Keep as deprecated |

**Tier 1 (In-Code):**

| Source | Target | Action |
|--------|--------|--------|
| `components/src/dynamo/planner/README.md` | Keep | Update link to `docs/components/planner/` |

### KVBM (972 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/kvbm/kvbm_intro.rst` | 69 | O | `docs/components/kvbm/README.md` | Convert | RST→MD |
| `docs/kvbm/kvbm_architecture.md` | 40 | O | `docs/components/kvbm/README.md` | Merge | Combine overviews |
| `docs/kvbm/kvbm_components.md` | 71 | O | `docs/components/kvbm/README.md` | Merge | Combine overviews |
| `docs/kvbm/kvbm_motivation.md` | 44 | O | `docs/components/kvbm/README.md` | Merge | Combine overviews |
| `docs/kvbm/kvbm_integrations.md` | 45 | G | `docs/components/kvbm/kvbm_guide.md` | Move | Integration instructions |
| `docs/kvbm/vllm-setup.md` | 195 | E | `docs/components/kvbm/kvbm_examples.md` | Merge | Combine examples |
| `docs/kvbm/trtllm-setup.md` | 223 | E | `docs/components/kvbm/kvbm_examples.md` | Merge | Combine examples |
| `docs/kvbm/kvbm_design_deepdive.md` | 262 | D | `docs/design_docs/kvbm_design.md` | Move | Design content |
| `docs/kvbm/kvbm_reading.md` | 23 | R | `docs/components/kvbm/kvbm_guide.md` | Merge | Append references |
| `docs/kvbm/kvbm_metrics_grafana.png` | — | — | `docs/components/kvbm/images/` | Move | Image asset |

**Tier 1 (In-Code):**

| Source | Target | Action |
|--------|--------|--------|
| `lib/bindings/kvbm/README.md` | Keep | Update link to `docs/components/kvbm/` |

### Frontend (2,991 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/frontends/kserve.md` | 99 | G | `docs/components/frontend/frontend_guide.md` | Move | KServe integration |
| `docs/frontends/openapi.json` | 2,892 | R | `docs/reference/api/openapi.json` | Move | API spec to reference |
| — | — | O | `docs/components/frontend/README.md` | Create | New overview needed |

**Tier 1 (In-Code):**

| Source | Target | Action |
|--------|--------|--------|
| `components/src/dynamo/http/README.md` | Create | Stub linking to `docs/components/frontend/` |

### Profiler (New)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| — | — | O | `docs/components/profiler/README.md` | Create | New overview |
| — | — | G | `docs/components/profiler/profiler_guide.md` | Create | New guide |

---

## 2. Backends

Backends remain at `docs/backends/<backend>/` with minimal structural changes.

### vLLM

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/backends/vllm/README.md` | — | O | Keep | — | No change |
| `docs/backends/vllm/disagg.md` | — | G | Keep | — | No change |
| `docs/backends/vllm/aggregated.md` | — | G | Keep | — | No change |
| `docs/backends/vllm/speculative_decoding.md` | — | G | `docs/features/speculative_decoding/` | Extract | Move to features |
| `docs/backends/vllm/LMCache_Integration.md` | — | G | `docs/integrations/lmcache/` | Extract | Move to integrations |

### SGLang

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/backends/sglang/README.md` | — | O | Keep | — | No change |
| `docs/backends/sglang/disagg.md` | — | G | Keep | — | No change |
| `docs/backends/sglang/aggregated.md` | — | G | Keep | — | No change |
| `docs/backends/sglang/sgl-hicache-example.md` | — | G | `docs/integrations/hicache/` | Extract | Move to integrations |

### TRT-LLM

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/backends/trtllm/README.md` | — | O | Keep | — | No change |
| `docs/backends/trtllm/disagg.md` | — | G | Keep | — | No change |
| `docs/backends/trtllm/aggregated.md` | — | G | Keep | — | No change |

---

## 3. Features

### Multimodal (1,644 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/multimodal/index.md` | 213 | O | `docs/features/multimodal/README.md` | Move | Rename to README |
| `docs/multimodal/vllm.md` | 522 | G | `docs/features/multimodal/multimodal_vllm.md` | Move | |
| `docs/multimodal/sglang.md` | 433 | G | `docs/features/multimodal/multimodal_sglang.md` | Move | |
| `docs/multimodal/trtllm.md` | 476 | G | `docs/features/multimodal/multimodal_trtllm.md` | Move | |

### Speculative Decoding (New)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/backends/vllm/speculative_decoding.md` | — | G | `docs/features/speculative_decoding/README.md` | Extract | From vLLM backend |

### Agents (183 lines)

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/agents/tool-calling.md` | 183 | G | `docs/features/agents/README.md` | Move | Agent/tool calling |

---

## 4. Integrations

### Extracted from Backends

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/backends/vllm/LMCache_Integration.md` | — | G | `docs/integrations/lmcache/README.md` | Extract | From vLLM |
| `docs/backends/sglang/sgl-hicache-example.md` | — | G | `docs/integrations/hicache/README.md` | Extract | From SGLang |

### NIXL

| Source | Lines | Type | Target | Action | Notes |
|--------|-------|------|--------|--------|-------|
| `docs/api/nixl_connect/*` | — | G | `docs/integrations/nixl/` | Move | Entire folder |

---

## Summary Statistics

### By Category

| Category | Files | Lines | Actions |
|----------|-------|-------|---------|
| Components | 18 | ~3,200 | Move, Merge, Convert, Create |
| Backends | 12 | varies | Extract only |
| Features | 6 | ~1,800 | Move, Extract |
| Integrations | 3+ | varies | Extract, Move |

### By Action Type

| Action | Count | Description |
|--------|-------|-------------|
| **Move** | ~15 | Simple relocation |
| **Merge** | ~8 | Combine multiple files |
| **Convert** | 2 | RST to Markdown |
| **Extract** | 4 | Pull from other files |
| **Create** | 4 | New content needed |

---

## Related Files

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Link update checklist and style guidelines
- [EXAMPLE_SKILL.md](EXAMPLE_SKILL.md) - Cursor skill for AI-assisted migration
- [README.md](README.md) - Template overview and selection guide
