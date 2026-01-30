---
orphan: true
---

# EXAMPLE: Planner Documentation Migration

This is an example prompt for migrating the Planner component documentation to the three-tier structure. Use this as a reference when creating migration prompts for other components.

---

## How to Use This Example

1. Copy this file and rename it for your component (e.g., `EXAMPLE_migration_router.md`)
2. Update the current files section with your component's existing docs
3. Update the target structure with your component's file names
4. Update the migration mapping tables with your component's content
5. Use the prompt with Claude 4.5 Opus Max mode

---

## Migration Prompt

```
You are migrating the Dynamo Planner documentation to a new three-tier structure. Work through this migration step by step, STOPPING at each checkpoint for user review before proceeding.

## Context

The Planner is an SLA-driven autoscaling component for Dynamo. The current documentation is scattered across multiple files with mixed concerns (architecture mixed with user guides, examples mixed with configuration).

## Current Files

Read these files to understand the current state:
- docs/planner/planner_intro.rst (83 lines) - RST entry point with feature matrix
- docs/planner/sla_planner_quickstart.md (522 lines) - DGDR workflow, steps, troubleshooting
- docs/planner/sla_planner.md (204 lines) - Architecture, load prediction, scaling algorithm
- docs/planner/load_planner.md (58 lines) - Deprecated, keep as-is

## Target Structure

Create this new structure:

docs/planner/
├── README.md              # Tier 2: Quick Start (80-100 lines)
├── planner_guide.md       # Tier 2: Guide (400-500 lines)
├── planner_examples.md    # Tier 2: Examples (300-400 lines)
└── load_planner.md        # Keep as deprecated archive

docs/design_docs/
└── planner_design.md      # Tier 3: Design (200-250 lines)

## Migration Mapping

### README.md (Quick Start)

| Section | Source | Est. Lines |
|---------|--------|------------|
| # Planner | planner_intro.rst overview | 10 |
| ## Feature Matrix | planner_intro.rst feature table (convert to MD) | 30 |
| ## Quick Start | sla_planner_quickstart.md Steps 1-4 (condensed) | 50 |
| ## Next Steps | New: links to guide, examples, design | 10 |

### planner_guide.md (Guide)

| Section | Source | Est. Lines |
|---------|--------|------------|
| ## Deployment | sla_planner_quickstart.md Prerequisites, Container Images, DGDR | 140 |
| ## Configuration | sla_planner_quickstart.md + sla_planner.md config sections | 140 |
| ## Integration | sla_planner.md Architecture, Virtual Deployment | 70 |
| ## Troubleshooting | sla_planner_quickstart.md Troubleshooting | 80 |

### planner_examples.md (Examples)

| Section | Source | Est. Lines |
|---------|--------|------------|
| ## Basic Examples | sla_planner_quickstart.md sample DGDRs | 140 |
| ## Kubernetes Examples | sla_planner_quickstart.md MoE, production examples | 130 |
| ## Advanced Examples | sla_planner.md warmup trace, Virtual Connector | 80 |

### planner_design.md (Design)

| Section | Source | Est. Lines |
|---------|--------|------------|
| ## Overview | sla_planner.md first paragraph | 20 |
| ## Architecture | sla_planner.md Architecture Overview + diagram | 30 |
| ## Load Prediction Models | sla_planner.md Load Prediction (algorithms) | 65 |
| ## Scaling Algorithm | sla_planner.md Scaling Algorithm (all 5 steps) | 85 |
| ## Performance/Future | New sections | 30 |

---

## Phase 1: Read and Analyze

1. Read all current docs for this component
2. Summarize what content exists where
3. Identify gaps or inconsistencies

**>>> STOP: Share your analysis. Ask if there are content priorities or known issues.**

---

## Phase 2: Create README.md

1. Convert any RST feature tables to Markdown
2. Write 2-3 sentence overview
3. Create condensed Quick Start
4. Add Next Steps links

**>>> STOP: Share README.md draft. Ask if feature matrix is current.**

---

## Phase 3: Create <component>_guide.md

1. Organize Deployment sections
2. Add Configuration sections (usage-focused)
3. Add Integration sections
4. Add Troubleshooting

**>>> STOP: Share guide draft. Ask if any config options are missing.**

---

## Phase 4: Create <component>_examples.md

1. Extract all code/YAML examples
2. Organize by complexity (basic → advanced)
3. Add comments to examples

**>>> STOP: Share examples draft. Ask if examples should be tested.**

---

## Phase 5: Create <component>_design.md

1. Extract architecture content
2. Extract algorithm details (formulas, pseudocode)
3. Preserve diagrams
4. Add Performance and Future Work sections

**>>> STOP: Share design draft. Ask if algorithms are current.**

---

## Phase 6 (Optional): Edit for Flow and Consistency

Review all four documents for:
- Terminology consistency
- Formatting consistency
- Cross-reference links
- Writing style (active voice, short sentences, BLUF)

**>>> STOP: Share edits. Ask if you should proceed.**

---

## Phase 7: Validate and Cleanup

1. Verify all original content is preserved
2. Check all links work
3. Delete original files (after approval)

Checklist:
- [ ] All content preserved
- [ ] No broken links
- [ ] Code examples correct
- [ ] Diagrams render

**>>> STOP: Share validation results. Ask for approval before deleting originals.**

---

## Constraints

- Preserve all existing content
- Keep code examples exactly as-is unless errors exist
- Preserve diagrams exactly
- Use consistent heading levels
- Add "See Also" links between docs
```

---

## Adapting This Example

When creating a migration prompt for another component:

1. **Update Context** - Describe what the component does
2. **Update Current Files** - List the actual files with line counts
3. **Update Target Structure** - Use correct component name
4. **Update Migration Mapping** - Map actual sections to new files
5. **Keep Phases** - The phase structure works for any component
