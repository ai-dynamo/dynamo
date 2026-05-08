# Agents working on this repo

This repo is **NVIDIA Dynamo** — a distributed LLM serving framework.
This file directs AI agents to the right entry point per task.

## Quick start by task

| If you're working on... | Read first |
|---|---|
| **DEP-XXXX Planner Plugin Architecture** (in-progress) | `DEP-XXXX_HANDOFF.md` |
| Planner plugin code conventions | `components/src/dynamo/planner/plugins/CLAUDE.md` |
| KVBM engine | `lib/kvbm-engine/CLAUDE.md` |
| KVBM kernels | `lib/kvbm-kernels/CLAUDE.md` |
| KVBM logical | `lib/kvbm-logical/CLAUDE.md` |
| Backend / engine ABC | `components/src/dynamo/common/backend/CLAUDE.md` |
| SGLang integration | `components/src/dynamo/sglang/CLAUDE.md`, `components/src/dynamo/sglang/AGENTS.md` |
| Filing a new DEP issue | `.claude/skills/dep-create/SKILL.md` |
| Checking DEP statuses | `.claude/skills/dep-status/SKILL.md` |
| Updating an existing DEP | `.claude/skills/dep-update/SKILL.md` |
| Tool-call parser for new model | `.claude/skills/tool-parser-generator/SKILL.md` |
| Maintaining Fern docs site | `.claude/skills/dynamo-docs/SKILL.md` |
| Filing a bug issue | `.claude/skills/gh-issue-bug/SKILL.md` |

## Repository conventions

- **Pre-commit hooks** run on every commit (Python format check + license
  header check); see `.cursor/rules/python-format-pre-commit.mdc`
- **License headers**: every new `.py` / `.proto` / shell script must
  start with the SPDX license header (see existing files for template)
- **Pytest markers**: new tests in any `components/src/dynamo/<area>/tests/`
  should declare appropriate markers (`pre_merge` / `gpu_0` / unit / etc.)
  to be auto-picked by CI; `.github/workflows/pr.yaml` shows which markers
  drive which jobs
- **No CI .yml edits** for new tests — use marker-based discovery

## DEP (Dynamo Enhancement Proposal) workflow

Major changes go through DEPs. Each DEP has:

- A GitHub issue with `dep:*` labels (status / area / type)
- A `DEP-XXXX_*.md` planning doc in repo root (Chinese for in-progress
  drafts; English when ready for upstream)
- Sub-PR detailed docs `DEP-XXXX_PR{N}_Detailed_zh.md`
- A `DEP-XXXX_HANDOFF.md` (when implementation is multi-session)

For DEP-XXXX specifically (Planner Plugin Architecture, in-progress):

```
DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md  ← main design (v11)
DEP-XXXX_Implementation_Breakdown_zh.md            ← 8-PR plan + cross-check
DEP-XXXX_PR{1..8}_Detailed_zh.md                   ← per-PR sub-task tables
DEP-XXXX_HANDOFF.md                                ← agent handoff entry
```

## Note on persistent context

If you (the agent) are continuing work that another agent started:

1. **Check the relevant `*_HANDOFF.md`** first if it exists — it has
   the implementation state, not just the design intent
2. **Read each affected PR doc's `修订历史` (revision history)** for
   decisions made during implementation that diverge from the original plan
3. **Verify clean baseline before adding code** by running the test
   commands listed in the HANDOFF doc

Conversation state from previous sessions is NOT preserved across CLI
invocations — all relevant context lives in the `*_HANDOFF.md` and
`修订历史` sections of the PR docs.
