<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Agent Skills

Portable, agentskills.io-compatible skills that teach AI agents how to use [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) correctly across its full lifecycle. Loaded by Claude Code, Cursor, Codex, and any other client that implements the [Agent Skills specification](https://agentskills.io/).

The `.agents/skills/` directory is the cross-client interop convention from [agentskills.io](https://agentskills.io/specification). In this repository `.claude/skills/` is a symlink to `.agents/skills/`, so Claude Code sees the same content under its native path.

These skills are mirrored daily to the public [NVIDIA/skills](https://github.com/NVIDIA/skills) catalog (registered as `components.d/dynamo.yml`).

---

## Skills

| Skill | Lifecycle stage | Owns | NV-ACES |
|---|---|---|---|
| [`dynamo-plan`](dynamo-plan/SKILL.md) | Plan | AIConfigurator workflow, DGDR `searchStrategy` choice (`rapid` vs `thorough`), SLA target framing, recipe selection | 92 |
| [`dynamo-optimize`](dynamo-optimize/SKILL.md) | Optimize | NVIDIA TensorRT Model Optimizer (`modelopt`) workflow, quantization technique selection (FP8 / NVFP4 / INT8 / AWQ), checkpoint validation | 94 |
| [`dynamo-serve`](dynamo-serve/SKILL.md) | Local run | Single-node workstation workflow via `python3 -m dynamo.<backend>`; per-backend launch-flag matrix | 92 |
| [`dynamo-deploy`](dynamo-deploy/SKILL.md) | Deploy | `dynamo-platform` Helm install, DGD + DGDR authoring, recipe-based deployment, conversion-webhook semantics, day-2 ops | 90 |
| [`dynamo-frontend`](dynamo-frontend/SKILL.md) | Request path | Frontend service (OpenAI endpoints `/v1/chat`, `/v1/completions`, `/v1/embeddings`, `/v1/realtime`), `DynamoModel` CR, multi-model serving, GAIE / kgateway / Istio integration | 92 |
| [`dynamo-troubleshoot`](dynamo-troubleshoot/SKILL.md) | Day-2 operations | Worker crashloops, inference 5xx, Planner stuck states, KV transfer fallback, conversion-webhook timeouts; NVBug-filing flow | 92 |
| [`dynamo-benchmark`](dynamo-benchmark/SKILL.md) | Benchmark | AIPerf workflow, in-tree `benchmarks/` suites, recipe-attached benchmark companions | 92 |

NV-ACES Tier 1 (Astra Continuous Evaluation for Skills) deterministic-scoring results from 2026-05-21. Average **92.1 / 100**; lowest **90**. All grades A- or A.

Two skills considered and deferred to a follow-up MR: `dynamo-install` (one-time cluster install of `dynamo-platform`) and `dynamo-upgrade` (release-to-release migration). Both have real workflows but require capturing the end-to-end command sequences before they're skill-ready.

---

## Philosophy

This skill set is built on a layered model:

| Layer | Lives | Refreshes |
|---|---|---|
| **Enduring architecture** | An internal architectural survey of the Dynamo source — every CRD, every Python entry point, every Rust crate, every container, every helm chart pin source, every companion project | When the architecture changes (new CRD, new lifecycle stage, directory rename). Rare. |
| **Per-release facts** | Each skill body: backend version pins, container image tags, NIXL / UCX refs, recipe list, known issues current to the release line | Every Dynamo release |
| **Skills (this directory)** | `.agents/skills/dynamo-<name>/SKILL.md` plus `references/` plus `scripts/` | Pulled by agent clients on demand via the progressive-disclosure model |
| **Public catalog mirror** | `NVIDIA/skills/skills/Dynamo/` | Daily automated sync from this directory; no human in the loop |

Each skill's `version` field in its frontmatter matches the Dynamo release line it targets. Users on a specific release line get the matching skill version. Multiple versions coexist when needed.

---

## Attribution

The structural conventions in these skills — the 4-phase workflow shape, the DESTRUCTIVE / MUTATING / SAFE command rubric, the Human-in-the-Loop behavioral contract, the `pass / fail / warn` and `check()` script patterns, the strict 6-element known-issues entry format, the frontmatter shape, the progressive disclosure block, the `references/` + `scripts/` subdirectory layout, and the NV-ACES evaluation infrastructure — were developed and battle-tested by the team behind the internal **`ai-infra-agent`** project. Several exemplar skills there established the patterns this skill set inherits:

- `nvidia-inference-stack` — the 4-phase workflow, `pass / fail / warn`
 script helper pattern, `check()` verification function, known-issues 6-element format, frontmatter shape.
- `nvidia-inference-ra-orchestrator` — the Human-in-the-Loop behavioral
 contract (present, then wait; no soft-language interpretation).
- `gpu-operator` — the DESTRUCTIVE / MUTATING / SAFE command-tier rubric
 with the explicit "you are responsible for the outcome" prompt.
- `KAI` — minimal-frontmatter exemplar; the directory-casing vs
 `name`-field-casing convention.

0% of the content in this directory is copy-paste from `ai-infra-agent`; ~100% of the structural rigor is inherited. The Dynamo work is a domain-specific application of an existing methodology, not a new convention. Where this skill set extends `ai-infra-agent`, the extensions are upstreamable to `ai-infra-agent`.

---

## Improvements on top of the `ai-infra-agent` methodology

A handful of extensions worth flagging — each is candidate for upstream adoption.

### 1. Architectural survey as a standalone input

A 633-line `DYNAMO_REPO_SURVEY.md` catalogs the Dynamo source's enduring shape: seven CRDs (DGD, DGDR, DCD, DGDSA, DynamoModel, DynamoCheckpoint, DynamoWorkerMetadata), twelve Python entry points (`dynamo.vllm`, `dynamo.trtllm`, `dynamo.sglang`, ...), twenty Rust workspace crates, container/Helm pin source files, companion projects (AIConfigurator, AIPerf, ModelOpt, NIXL, Grove, KAI, ModelExpress). Skills cite the survey by section anchor rather than embedding architectural facts inline. One survey backs many skills and refreshes independently when the product re-shapes.

### 2. Citation manifest with `VERIFIED` / `PENDING` / `UNVERIFIED` states

A 130-row `citations.md` ties every load-bearing claim in every skill body, references file, and authoring guide to a verifiable source. Three states: `VERIFIED` (file exists, snippet matches verbatim), `PENDING` (row added but not yet checked), `UNVERIFIED` (source unreachable; body prose forbidden until the row is promoted or removed). `UNVERIFIED` claims fail the pre-MR audit. The row-ID-by-section model (A: house style, B: public catalog, C: DGDR, D: known-issue patterns, E: self-evaluation, F: survey-backed Dynamo facts) lets multiple skills reuse the same citations.

### 3. Layered master-derivative publication pipeline

`scripts/derive-public.sh` (241 lines, shellcheck-clean) mechanically derives public artifacts from the rigorous internal master. Two targets: this `.agents/skills/` directory (full rigor) and the public `NVIDIA/skills` catalog (mirrored automatically; no transform needed because the catalog sync mirrors source repos verbatim). The script emits `MANUAL_REVIEW` markers for the editorial gaps it can't resolve mechanically and exits non-zero, gating publication on human review.

### 4. Lifecycle-stage decomposition

Seven peer skills covering the user journey end-to-end, with cross-skill referencing tables in each skill instead of a central meta-orchestrator. `ai-infra-agent`'s `nvidia-inference-stack` bundled install + deploy + benchmark into one large skill; `nvidia-inference-ra-orchestrator` inverts that into a top-down router. Flat peer skills scale without a bottleneck when the skill count grows.

### 5. Per-release skill versioning

Each skill's `version` field is tied to the Dynamo release line, not arbitrary semver. The refresh workflow per release is codified (see "How we update and version" below). `ai-infra-agent` skills don't bump per product release.

### 6. NV-ACES standard-headers extension

Adding `## Workflow` (anchoring the 4-phase block under a header the evaluator matches), `## Available Scripts` (table with `run_script()` example invocations), `## Prerequisites`, `## Limitations`, and `## Troubleshooting` lifted average NV-ACES Tier 1 from 76.3 (Grade C) to 92.1 (Grade A-) — a 15.8-point lift. The patch is non-invasive: existing skill content stays; the new headers wrap it in the evaluator's expected vocabulary. `ai-infra-agent` could adopt the same patch and get the same lift on its skills.

### 7. `run_script()` protocol surfacing

The agentskills.io spec defines `run_script("scripts/<name>.sh", args=[...])` as the canonical agent-side invocation of bundled scripts. Every skill in this directory pairs the `bash scripts/<name>.sh ...` form (for humans reading the docs) with the `run_script(...)` form (for agent consumers). NV-ACES Tier 1 scans for the latter.

### 8. Phase-shape variance for non-install workflows

`ai-infra-agent`'s 4-phase shape is Pre-Check → Install → Validate → Deploy — install-oriented. For `dynamo-troubleshoot` (a day-2 skill, not install) we substituted Triage → Inspect → Diagnose → Remediate while keeping the four-phase contract. The extension is documented in the authoring guide so other day-2 skills can follow the same pattern.

### 9. Survey-driven feature-naming check

Before any skill names a tool or surface, the survey verifies it exists. Caught `dynamo-run` early: that CLI does not exist in the Dynamo source (the binary was never built), but generic agent documentation often refers to it. Naming a non-existent tool would break the skill the first time an agent tried to invoke it. The principle generalizes: skills cite tools and surfaces against a verified inventory, not from training-data memory.

### 10. XML-tag pitfall in YAML descriptions

Angle-bracket placeholders (`<backend>`, `<n>`) inside YAML folded- scalar `description` fields get parsed as XML by NV-ACES and fail with an error. Hit on `dynamo-serve` and `dynamo-deploy`; fixed; documented. Future skill authors avoid this.

---

## How we update and version

### What lives where

| Layer | Refresh trigger | Cadence |
|---|---|---|
| `DYNAMO_REPO_SURVEY.md` (enduring architecture) | New CRD, new lifecycle stage, directory rename | When the architecture shifts — rare |
| Skill body (per-release facts) | New Dynamo release — backend pins, container tags, NIXL refs, recipe list, known issues | Every release line |
| `citations.md` `Checked` column | Source files re-read against the new release | Every release bump |
| `NVIDIA/skills` catalog mirror | Automated daily sync from this `.agents/skills/` directory | Daily |

### Per-release refresh procedure

1. **Fetch the release branch.**
 ```bash
 git fetch origin release/<X.Y.Z>
 ```
2. **Re-derive per-release facts.** For each skill that needs the value,
 run `git show origin/release/<X.Y.Z>:<source-file>` and embed the result. Typical touchpoints:
 - `Cargo.toml` workspace version → skill `version` field
 - `pyproject.toml` `[project.optional-dependencies]` → backend pin
 tables (vLLM, TRT-LLM, SGLang)
 - `container/context.yaml` `nixl_ref` / `nixl_ucx_ref` → NIXL pin
 tables
 - `deploy/helm/charts/platform/values.yaml` → etcd / NATS / operator
 image tags
 - `recipes/` directory listing → recipe set in `dynamo-deploy` and
 `dynamo-benchmark`
3. **Refresh known-issues** against the active QA tracker for the
 release line. Resolved issues fall off the signature catalog; new RC cherry-pick fixes get added.
4. **Bump the `version` field** in every touched skill's frontmatter.
 Mandatory if any per-release fact changed.
5. **Re-run the pre-MR checklist.** YAML frontmatter parses, every claim
 cites a `VERIFIED` row, every cluster-mutating command appears in a DESTRUCTIVE / MUTATING table, every script passes `shellcheck`, cross-link audit clean.
6. **Re-run `astra-skill-eval` locally** against each touched skill.
 Score should stay ≥ 90 (current floor); fix any new findings.
7. **Open the PR** to this repository. The `NVIDIA/skills` catalog picks
 up the change in the next daily sync — no second PR per release.

### When the survey itself refreshes

Most release bumps touch the skill body and not the survey. The survey refreshes when:

- A CRD is added, removed, or has its served/storage versions changed
 (e.g., `DynamoModel` promoted from v1alpha1 to v1beta1).
- A new lifecycle stage appears.
- A top-level directory is renamed or restructured (`components/`,
 `lib/`, `deploy/`, `examples/`).
- A companion project's role changes (e.g., AIPerf replaces GenAI-Perf).

Survey-refresh PRs touch many skills downstream (citations need updating). They're rare but worth flagging as refactor PRs rather than routine release bumps.

### Cross-release skill pattern

Skills meant to span multiple release lines gate per-release facts behind a release variable. The skill body reads the user's target release and substitutes the relevant pin. The primary lifecycle skills in this directory pin explicitly; the cross-release pattern is reserved for future adoption skills.

---

## Quality bar

The pre-MR checklist that gates publication:

- YAML frontmatter parses; required fields (`name`, `description`,
 `version`, `author`, `tags`, `tools`) present.
- Every load-bearing claim in every SKILL.md and `references/*.md`
 cites a `VERIFIED` row in the citation manifest.
- Every cluster-mutating command appears in a DESTRUCTIVE / MUTATING
 table (or a SAFE block); the human-in-the-loop contract applies to every decision point.
- Every script passes `shellcheck` with no warnings.
- Every script defaults to a non-destructive mode (e.g.
 `kubectl apply --dry-run=server` rather than `kubectl apply`).
- Cross-link audit clean — every `references/*.md` and `scripts/*.sh`
 link in every SKILL.md resolves on disk.
- Local `astra-skill-eval evaluate <skill> --static` ≥ 90 with no
 errors.

---

## Compatibility

- **Agent Skills specification:** [agentskills.io/specification](https://agentskills.io/specification).
 `name` and `description` frontmatter fields are required; `SKILL.md` is required at each skill directory root. These skills also use the optional `version`, `author`, `tags`, and `tools` fields plus the `## Workflow` / `## Available Scripts` / `## Prerequisites` / `## Limitations` / `## Troubleshooting` section headers (per the authoring guide that backs this set).
- **Clients:** Cursor walks `.agents/skills/` natively; Claude Code walks `.claude/skills/` (a symlink to `.agents/skills/` in this repo); Codex walks `.codex/skills/`. The cross-client convention is `.agents/skills/`.
- **Mirror:** `NVIDIA/skills` syncs `.agents/skills/` daily via
 `components.d/dynamo.yml`.

## License

This directory is dual-licensed under Apache 2.0 (code, scripts) and CC-BY-4.0 (skill content). See [LICENSE](../../LICENSE).

## Security

Security issues with the skills themselves (script injection, instruction exfiltration) follow the Dynamo security reporting policy: see [SECURITY.md](../../SECURITY.md).
