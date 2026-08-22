<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo — Agent Guide

## Overview

Dynamo is NVIDIA's open-source, datacenter-scale distributed inference framework. It is
the orchestration layer **above** inference engines (SGLang, TensorRT-LLM, vLLM), not a
replacement for them: it turns a cluster of GPUs into one coordinated inference system.
Core capabilities are disaggregated prefill/decode serving, KV-aware routing, multi-tier
KV cache management (KVBM: GPU → CPU → SSD → remote), SLA-driven autoscaling (Planner),
in-flight fault tolerance, and a Kubernetes operator for deployment.

The stack is deliberately layered and large. A **Rust core** (a Cargo workspace of
twenty-plus crates, mostly under `lib/`) holds the runtime, LLM, routing, and
KV-block-manager engines. A **Python
extensibility layer** (the `ai-dynamo` wheel, bound to the Rust core through PyO3/maturin)
holds the frontend, backends, planner, and profiler. A **Kubernetes layer** (`deploy/`)
holds the operator, Helm charts, and gateway integration. Treat any change that crosses
these boundaries as non-trivial. Dynamo also sits inside a wider `ai-dynamo` ecosystem of
sibling repos (below) that it integrates with rather than vendors.

## Skills

Skills live canonically in `.agents/skills/`; `skills/` and `.claude/skills/` are symlinks
to it — edit only the canonical copy. Reach for the right group first:

**For developing Dynamo:**

- `debug-session` — structured bug investigation with a persistent worklog
- `dep-create` — create or update Dynamo Enhancement Proposals as GitHub issues
- `dep-status` — check DEP status and list DEPs by lifecycle state or area
- `dep-update` — advance DEP lifecycle: triage, PIC assignment, review, approval
- `dynamo-clone-hotpath-audit` — audit Rust hot-path `.clone()` calls
- `dynamo-docs` — Fern docs-site content per the style guide
- `dynamo-frontend-benchmark` — benchmark/profile the frontend against mock workers
- `fern-components` — Fern MDX component library and usage guidance
- `fern-navigation` — Fern navigation and site-structure configuration guidance
- `dynamo-kv-replay-parity` — validate offline KV replay parity and performance
- `dynamo-agent-harness` — drive persistent Claude Code, Codex, or OpenCode sessions through Dynamo over ACP
- `graham-code-review` — strict Rust/systems review in Graham King's style
- `issue-create` — open upstream issues using the repository's GitHub issue forms
- `pr-create` — open an upstream pull request with compliant title, body, and DCO checks
- `pr-monitor` — CI health check, failure root-cause, and skip analysis
- `visual-review` — interactive HTML code-review dashboards with diagrams and annotated diffs

**For deploying and operating Dynamo:**

- `synthesize-user-workload` — interview the user, capture their confirmed baseline DGD, and create the canonical workload contract
- `author-baseline-dgd` — draft a baseline DGD from interview requirements when no recipe matches, for the user's confirmation
- `consult-perf-knowledge` — select one evidence-backed optimization proposal and write its reasoning record
- `create-optimization-hypothesis` — materialize a performance consultation as a challenger-ready DGD draft
- `perform-adversarial-review` — challenge a generated DGD candidate before it consumes GPU time
- `deploy-dynamo-recipe` — deploy one assigned Kubernetes DGD and verify it with an API smoke test
- `configure-aiperf-benchmark` — freeze and render a comparable AIPerf workload for a deployed candidate
- `run-aiperf-benchmark` — execute and collect one run-scoped AIPerf Kubernetes benchmark
- `analyze-aiperf-results` — validate AIPerf evidence, evaluate SLOs, and compare valid same-series runs
- `dynamo-router-starter` — start/patch router modes with smoke checks
- `dynamo-interconnect-check` — validate NIXL/UCX/NCCL readiness for disaggregation
- `troubleshoot-dynamo` — diagnose failed or unhealthy deployments

**Adding a skill:** the folder name must equal the frontmatter `name` (kebab-case); the
`description` is third person, states what the skill does and when to use it, and is at
most 1024 characters; include `license: Apache-2.0` and a `metadata:` block with `author`
and `tags`. List the skill in this section — the index must match `.agents/skills/`
exactly. All of this is enforced by `scripts/validate_skills.py` (pre-commit hook
`validate-skills`). Changes under `.agents/skills/` are also validated by NVSkills CI —
a maintainer comments `/nvskills-ci` on the PR.

## Improving These Instructions

If these skills or instructions misled you, blocked you, or contradicted what you verified live, prepare an issue for
this repository with the `agent-reported` label and ask your operator to approve filing it — filing is an external
write and requires operator consent. Rules:

1. Search existing `agent-reported` issues first; propose commenting on a duplicate instead of filing a new one.
2. Prepare at most one issue per optimization session; batch findings into it.
3. Identify yourself as an AI agent, including your driver model and the skills commit you were running.
4. Sanitize completely: no user workload details, traffic numbers, cluster or namespace names, company names, or
   credentials. Describe the instruction gap, not the engagement. Show the operator the full draft before filing.

## Optimization Role Dispatch

When the first user message starts a new Dynamo recipe optimization run, FIRST read
`agent-docs/guides/optimization/optimize-loop.md` end to end - the individual SKILL.md files are auto-discoverable,
but the loop's sequencing, state machine, and stopping rules live only in that guide - then dispatch
`user_interviewer` before any other specialized role. It must invoke `synthesize-user-workload` and produce a validated
`<EXP_ROOT>/user_workload.yaml` plus an immutable `<EXP_ROOT>/inputs/user_provided_dgd.yaml` copied from the baseline the
user supplied or explicitly confirmed. Do not dispatch `recipe_deployer`, `perf_analyzer`, `hypothesis_generator`, or
`hypothesis_challenger` until both exact paths and SHA256 values are available. Pass both inputs directly to
`recipe_deployer`; pass the same immutable workload path and hash to every later role. Do not insert a recipe
exploration or selection step after the interview: the baseline-source ladder
(`agents/user-interviewer/AGENTS.md`) is the only place selection or authoring happens, always with the user's
explicit confirmation, and always before the loop starts.

## Long-Running Runs And Harness Compatibility

An optimization loop is long-running, unattended work. Know which harness you are in: in a SINGLE-SHOT harness
(headless `-p`/print mode, one-turn API calls), background-job completion notifications can never reach you - the
session is gone when your turn ends. There, poll synchronously with bounded loops and never park the engagement on
a wake-up you cannot receive; parking is only valid where the harness can re-invoke you (interactive sessions, goal
mode). An interactive harness ends its turn whenever the agent stops
calling tools — a turn that ends on narrated intent ("now I'll test disagg") silently stalls the loop until a human
notices. Two rules:

1. **Operators: launch unattended runs inside your harness's goal mode.** Goal mode is an operator action at launch,
   not something these instructions can enable mid-run. On Codex CLI, wrap the run in `/goal` with a token budget. On
   Claude Code (v2.1.139+), wrap it in `/goal`; its completion condition is model-evaluated and may include a bound
   such as "or stop after N turns" as part of the condition text (a soft limit, not a hard budget). A validated
   condition template: "Test every lever family that is testable within the authorized budget. Never stop because a
   report exists. Parked on pending asks with nothing else testable is a valid pause, not completion. Valid stops: an
   operator-granted stop-request, the authorized budget exhausted, access lost, or operator interrupt." Always name
   the budget (GPU-hours, wall-clock, failed-deployment limit) in the condition; a bare "never stop" silently relies
   on credential expiry as its budget. Tell the operator at the START of any optimization
   engagement — not only when they say "unattended" — that this is long-running work and how to arm goal mode; the
   user-interviewer's contract handoff is the natural moment. Arm goal mode only AFTER the contract questions are
   answered: a goal hook armed while questions are outstanding forces the run past them onto its own defaults. The template's parked-on-asks pause assumes a
   reachable operator: for runs where the operator will be away, instruct the agent not to park on asks (asks are
   logged and the loop continues) and keep only the hard stops. Blocking question tools suspend the turn BEFORE the
   goal hook can evaluate, so one blocking question can hang an unattended run for hours; harnesses that support
   tool restrictions should disallow blocking question tools in goal mode.
2. **Never end a turn on narrated intent during a loop.** Either perform the next step in the same turn, launch it as
   background work that will re-invoke you, or return the specific blocking question you need answered.

**Harness tiers.** These roles and skills are developed and tested on Claude Code and Codex CLI. Isolated role
configurations currently ship for Codex only (`.codex/agents/*.toml`); on Claude Code the roles run in-context within
one session (no `.claude/agents/` configurations yet), so adversarial review there is same-context review, not an
independent reviewer. The skills follow the Agent Skills open standard and load on other compliant harnesses (for
example, OpenCode includes `.agents/skills/` among its standard skill search paths), with the same in-context role
caveat plus two more degradations: no native goal mode (run lights-out sessions under an external loop), and every
rule in this pack is prompt-enforced, so discipline depends on the driver model. If you hit an instruction gap on any
harness, prepare a sanitized issue describing the gap and ask your operator to approve filing it on this repository.

## Ecosystem

Sibling repositories this repo integrates with:

| Repo | Role |
|------|------|
| [NIXL](https://github.com/ai-dynamo/nixl) | High-throughput inference data-transfer library (KV-cache transfer over RDMA/NVLink) that underpins disaggregated serving |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | Benchmarking and load-generation tool used by the benchmarking guides |
| [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) | Simulates thousands of deployment configs to find an optimal serving config before spending GPU-hours |
| [ModelExpress](https://github.com/ai-dynamo/modelexpress) | Streams model weights GPU-to-GPU via NIXL for fast replica cold-start |
| [Grove](https://github.com/ai-dynamo/grove) | Kubernetes operator for topology-aware gang scheduling |

## Repository Map

| Path | Contents |
|------|----------|
| `lib/` | Rust workspace crates: `runtime`, `llm`, `kv-router`, `kvbm-*`, `mocker`, and more (see the root [`Cargo.toml`](Cargo.toml) `[workspace] members`), plus `bindings/python` — the PyO3 extension crate, built via maturin and deliberately excluded from the workspace |
| `components/src/dynamo/` | Python packages: `frontend`, `planner`, `router`, `vllm`/`sglang`/`trtllm` backends, `mocker`, `profiler`, and more |
| `deploy/` | Kubernetes `operator`, Helm charts, `inference-gateway` ext-proc, `observability` |
| `container/` | Dockerfiles and build scripts for runtime and dev images |
| `docs/`, `fern/` | Documentation sources and the Fern docs-site config — read [`docs/AGENTS.md`](docs/fern/AGENTS.md) before editing |
| `examples/`, `recipes/` | Runnable examples and deployment recipes — also covered by [`docs/AGENTS.md`](docs/fern/AGENTS.md) |
| `benchmarks/`, `tests/` | Benchmark harnesses and the top-level pytest suite |
| `.ai/` | Agent topic guidelines: `bash-launch-guidelines.md`, `ci-guidelines.md`, `linear-ticket-refs.md`, `pytest-guidelines.md`, `python-guidelines.md`, `test-model-size-guardrails.md` |
| `.agents/skills/` | Agent skills (see [Skills](#skills)) |

## Build

System prerequisites (Rust toolchain, `uv`, system libraries) and the VS Code / Cursor
devcontainer are covered in [`docs/contribution-guide.md`](docs/fern/pages/community/contributing/overview.md).

Python dev build (bindings + wheel, editable):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
cd lib/bindings/python && maturin develop --uv && cd -
uv pip install -e lib/gpu_memory_service
uv pip install -e .
python3 -m dynamo.frontend --help   # verify
```

Rust-only:

```bash
cargo build                 # whole workspace
cargo build -p dynamo-llm   # one crate
```

## Test

```bash
cargo test                  # Rust
pytest -m unit tests/       # Python unit tests
```

Markers are strict (`--strict-markers`); the full marker list lives in
[`pyproject.toml`](pyproject.toml) `[tool.pytest.ini_options]`, including GPU gating
(`gpu_0` … `gpu_8`). Read [`.ai/pytest-guidelines.md`](.ai/pytest-guidelines.md) and
[`.ai/test-model-size-guardrails.md`](.ai/test-model-size-guardrails.md) before writing
tests.

## Lint

```bash
pre-commit run --all-files            # all hooks (run `pre-commit install` first; it also installs the DCO commit-msg hook)
cargo fmt --all && cargo clippy --workspace
```

## PR and Commit Conventions

- Keep changes focused and reviewable.
- Use Conventional Commit PR titles: `type(scope): summary`. Accepted types:
  `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `perf`, `chore`, `revert`,
  `style`, and `build`.
- PR descriptions must include `Summary` and `Validation`.
- Sign every commit with DCO: `git commit -s`.
- Do not hand-edit the root `CODEOWNERS` — it is generated. To change review
  routing, edit `.github/codeowners/areas.yaml` and regenerate; CI gates 100%
  coverage and `CODEOWNERS`↔`areas.yaml` drift. See
  `.github/codeowners/README.md` (use `who_owns.py` to check who reviews a path).
- Full CI on a PR runs only after a maintainer comments `/ok to test <sha>` with the short
  SHA of the latest commit; copy-pr-bot then creates the `pull-request/N` branch that
  triggers it. Fix failures before requesting human review.
- Architecture changes require a Dynamo Enhancement Proposal (DEP), filed as a GitHub
  issue on `ai-dynamo/dynamo` with `dep:*` labels (the `dep-create` skill automates this).

See [`docs/contribution-guide.md`](docs/fern/pages/community/contributing/overview.md) for the full workflow
(issue sizing, CODEOWNERS, review process).

## Docs, Examples, Recipes

Any change under `docs/`, `examples/`, or `recipes/` must follow
[`docs/AGENTS.md`](docs/fern/AGENTS.md) and the
[documentation style guide](docs/fern/pages/community/contributing/documentation/documentation-style-guide.md): SPDX headers, Fern
frontmatter (no body `# H1`), GitHub-style admonitions, and backend casing
(vLLM / SGLang / TensorRT-LLM). The deterministic subset is enforced pre-merge.
