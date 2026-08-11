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
- `pr-monitor` — CI health check, failure root-cause, and skip analysis
- `visual-review` — interactive HTML code-review dashboards with diagrams and annotated diffs

**For deploying and operating Dynamo:**

- `dynamo-recipe-runner` — select, patch, and deploy Kubernetes recipes
- `dynamo-router-starter` — start/patch router modes with smoke checks
- `dynamo-interconnect-check` — validate NIXL/UCX/NCCL readiness for disaggregation
- `dynamo-troubleshoot` — diagnose failed or unhealthy deployments

**Adding a skill:** the folder name must equal the frontmatter `name` (kebab-case); the
`description` is third person, states what the skill does and when to use it, and is at
most 1024 characters; include `license: Apache-2.0` and a `metadata:` block with `author`
and `tags`. List the skill in this section — the index must match `.agents/skills/`
exactly. All of this is enforced by `scripts/validate_skills.py` (pre-commit hook
`validate-skills`). Changes under `.agents/skills/` are also validated by NVSkills CI —
a maintainer comments `/nvskills-ci` on the PR.

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

## Cursor Cloud specific instructions

The Cloud VM is **CPU-only (no GPU)** with **no HuggingFace Hub egress** (downloads fail
with connection reset). GitHub and PyPI egress do work. The build is pre-done in the VM
snapshot; the startup update script only refreshes the two editable Python installs
(`ai-dynamo` and `lib/gpu_memory_service`) into `.venv` — it does **not** rebuild the Rust
bindings. Activate the env with `source .venv/bin/activate` before running anything.

- **Rust changes are NOT hot-reloaded into Python.** After editing anything the PyO3
  bindings depend on (`lib/bindings/python` or any workspace crate under `lib/`),
  rebuild with `cd lib/bindings/python && maturin develop --uv`. A plain `uv pip install -e .`
  will not pick up Rust changes. First build is ~3–5 min; incremental is faster.
- **`cc`/`c++` are pointed at gcc/g++** (via `update-alternatives`, baked into the
  snapshot). This is required because the default clang can't find libstdc++ headers when
  cc-rs passes `--target=...`, which breaks C/C++ deps like the vendored ZeroMQ. If you
  rebuild in a context where those alternatives aren't set, prefix builds with
  `CC=gcc CXX=g++`.
- **Run the product locally without GPU/etcd/NATS** using the mocker worker + file-based
  discovery. HF Hub is blocked, so point `--model-path` at a local sample-model dir that
  contains a `chat_template` (the frontend refuses to materialize a model without one).
  `lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct` works; `TinyLlama_v1.1`
  does **not** (no `chat_template`). Do **not** pass `--model-name` — let the model id
  default to the path so the frontend loads the tokenizer from disk instead of trying HF.
  Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Example (two shells):
  `python -m dynamo.frontend --http-port 8000 --discovery-backend file` and
  `python -m dynamo.mocker --model-path lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct --discovery-backend file`,
  then `curl localhost:8000/v1/chat/completions`. The mocker simulates scheduling/KV/timing
  and emits synthetic token IDs, so response `content` is `null`/garbage — the meaningful
  signal is the request lifecycle and token/latency metrics, not the text. Real inference
  (vLLM/SGLang/TensorRT-LLM) and anything needing Kubernetes require a GPU/cluster and are
  out of scope here.
- **Unit tests** (`pytest -m unit tests/`, see [Test](#test)) need
  `--continue-on-collection-errors` on CPU-only: `tests/serve/test_{vllm,sglang,trtllm}.py`
  and `tests/fault_tolerance/test_canary_rank_pause.py` fail to import (`torch`/
  `tensorrt_llm` absent). Three unit tests fail by design in this env and are not setup
  problems: `test_kvbm_wheel_exists`/`test_kvbm_imports` (expect a prebuilt KVBM wheel in
  `/opt/dynamo/wheelhouse/`, only present in the container image) and
  `test_aiconfigurator_consistency` (optional `aiconfigurator` sibling package not
  installed).
- **Lint**: `cargo fmt --all -- --check` and `cargo clippy --workspace` work directly;
  `pre-commit` hooks fetch their environments from GitHub on first run (egress works).
