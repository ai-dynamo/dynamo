# Documentation Templates

Templates for creating consistent Dynamo documentation.

## Directory Hierarchy

### Components (Router, Planner, KVBM, Frontend, Profiler)

```
components/src/dynamo/<component>/
└── README.md                         ← incode_readme.md (Tier 1)

docs/<component>/
├── README.md                         ← component_readme.md (Tier 2)
├── <component>_guide.md              ← component_guide.md (Tier 2)
└── <component>_examples.md           ← component_examples.md (Tier 2)

docs/design_docs/
└── <component>_design.md             ← component_design.md (Tier 3)
```

### Backends (vLLM, SGLang, TRT-LLM)

```
components/src/dynamo/<backend>/
└── README.md                         ← incode_readme.md (Tier 1)

docs/backends/
├── README.md                         ← Backend comparison (exists)
└── <backend>/
    ├── README.md                     ← backend_readme.md (Tier 2)
    └── <backend>_guide.md            ← backend_guide.md (Tier 2)
```

### Features (Multimodal, LoRA, Speculative Decoding)

```
docs/features/<feature>/
├── README.md                         ← feature_readme.md (Tier 2)
├── <feature>_vllm.md                 ← feature_backend.md (Tier 2)
├── <feature>_sglang.md               ← feature_backend.md (Tier 2)
└── <feature>_trtllm.md               ← feature_backend.md (Tier 2)
```

### Integrations (LMCache, HiCache, NIXL)

```
docs/integrations/<integration>/
├── README.md                         ← integration_readme.md (Tier 2)
├── <integration>_setup.md            ← (custom)
└── <integration>_<backend>.md        ← (custom)
```

## Three-Tier Pattern

| Tier | Purpose | Audience | Location |
|------|---------|----------|----------|
| **Tier 1** | Redirect stub (5 lines) | Developers browsing code | `components/src/dynamo/<name>/README.md` |
| **Tier 2** | User documentation | Users, operators | `docs/<category>/<name>/` |
| **Tier 3** | Design documentation | Contributors | `docs/design_docs/<name>_design.md` |

## Template Selection

| What you're documenting | Templates to use |
|------------------------|------------------|
| New component | `incode_readme.md` + `component_*.md` (all 4) |
| New backend | `incode_readme.md` + `backend_*.md` (both) |
| New feature | `feature_readme.md` + `feature_backend.md` (per backend) |
| New integration | `integration_readme.md` |
| Migrating existing docs | Use the template matching your target file |

## Usage

1. Identify which category your documentation belongs to (component, backend, feature, integration)
2. Create the directory structure shown above
3. Copy templates to the correct locations with correct filenames
4. Replace all `<placeholders>` with actual values
5. Replace `<!-- comments -->` with actual content
6. Remove sections that don't apply

## Migrating Existing Docs

For migrating existing documentation to the new structure:

1. See [EXAMPLE_migration_planner.md](EXAMPLE_migration_planner.md) for a complete migration prompt
2. Copy and adapt the example for your component
3. Use with Claude 4.5 Opus Max mode for best results
4. Follow the phased approach with STOP points for review
