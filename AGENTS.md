<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo — Agent Guide

[`README.md`](README.md) explains what Dynamo is and what it supports.

## Repository Map

| Path | Contents |
|------|----------|
| `lib/` | Rust crates, including `runtime`, `llm`, `kv-router`, `kvbm-*`, `mocker`, `sidecar/*` (engine gRPC sidecars), and `bindings/python` (the PyO3 extension, built with maturin, outside the Cargo workspace) |
| `components/src/dynamo/` | Python packages: `frontend`, `planner`, `router`, `vllm`/`sglang`/`trtllm`/`tokenspeed` backends, `mocker`, `profiler`, and more |
| `deploy/` | Kubernetes `operator`, Helm charts, `inference-gateway` ext-proc, `observability` |
| `container/` | Dockerfiles and build scripts for runtime and dev images |
| `docs/fern/` | Fern docs site; pages are in `pages/` |
| `examples/`, `recipes/` | Runnable examples and deployment recipes |
| `benchmarks/`, `tests/` | Benchmark harnesses and the top-level pytest suite |
| `.ai/` | Topic guidelines for agents (pytest, Python, CI, launch scripts, AISimulate); read the matching one before writing that kind of code |
| `.agents/skills/` | Agent skills (see [Skills](#skills)) |
| `agents/`, `agent-docs/` | Role contracts, guides, and rules for the recipe optimization loop |

## Agent Instruction Files

- `AGENTS.md` is the canonical and only source of agent instructions at every
  repository scope.
- Every `AGENTS.md` must have a sibling `CLAUDE.md` containing exactly
  `@AGENTS.md` so Claude loads the same instructions.
- Do not put independent instructions in `CLAUDE.md`, and do not make
  `AGENTS.md` a symlink to `CLAUDE.md`.
- Add, move, or remove each `AGENTS.md` and `CLAUDE.md` pair together.

## Component Guides

This file is the whole-repository view. Each major tier carries its own agent
guide with the crate or package map and the rules that are local to it, and many
individual crates and modules nest a further one inside that. Read the guide for
the tier you are working in rather than inferring its layout from here:

| Tier | Guide |
|------|-------|
| Rust workspace | [`lib/AGENTS.md`](lib/AGENTS.md), and [`lib/runtime/AGENTS.md`](lib/runtime/AGENTS.md) for the runtime crate |
| Python components | [`components/src/dynamo/AGENTS.md`](components/src/dynamo/AGENTS.md) |
| Kubernetes | [`deploy/AGENTS.md`](deploy/AGENTS.md), and [`deploy/operator/AGENTS.md`](deploy/operator/AGENTS.md) for the operator |
| Docs, examples, recipes | [`docs/fern/AGENTS.md`](docs/fern/AGENTS.md), [`recipes/AGENTS.md`](recipes/AGENTS.md) |

Follow the [Agent Instruction Files](#agent-instruction-files) convention at each
scope. When a nested guide and a broader one both apply, the nested one wins.

## Concepts

The architecture is documented once, under
[`docs/fern/pages/developer-guide/knowledge-base/`](docs/fern/pages/developer-guide/knowledge-base/).
Read it there rather than reconstructing it from source:

- [Architecture](docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md) — request flow and the runtime/namespace/component/endpoint model
- [Disaggregated serving](docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md) — splitting prefill from decode
- [Routing concepts](docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md) — KV-aware routing
- [KVBM configuration reference](docs/fern/pages/reference/components/kvbm-configuration.mdx) — multi-tier KV cache management

## Ecosystem

Dynamo serves [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), and
[TokenSpeed](https://github.com/lightseekorg/tokenspeed). Related projects:
[NIXL](https://github.com/ai-dynamo/nixl) (KV-cache transfer),
[AIPerf](https://github.com/ai-dynamo/aiperf) (load generation),
[AISimulate](https://pypi.org/project/aisimulate/) (offline configuration search),
[ModelExpress](https://github.com/ai-dynamo/modelexpress) (GPU-to-GPU weight streaming), and
[Grove](https://github.com/ai-dynamo/grove) (Kubernetes gang scheduling).

## Skills

Skills live canonically in `.agents/skills/`; `skills/` and `.claude/skills/` are symlinks
to it — edit only the canonical copy.

**Build, debug, and measure:**

- `debug-session` — investigate a bug with a persistent worklog: capture the environment, rebuild, reproduce
- `dynamo-frontend-benchmark` — benchmark and profile the frontend against `dynamo.mocker` workers
- `dynamo-kv-replay-parity` — byte-parity and paired performance campaigns for offline KV-aware replay
- `dynamo-clone-hotpath-audit` — audit Rust hot-path `.clone()` calls
- `dynamo-agent-harness` — drive Claude Code, Codex, or OpenCode sessions through a Dynamo endpoint over ACP

**Pull requests, review, and CI:**

- `pr-checklist` — check a PR for design fit, tests, code quality, and description, when making or reviewing one
- `dynamo-perf-best-practices` — check hot-path changes in a PR against Dynamo's performance practices; pr-checklist runs it
- `graham-code-review` — strict Rust/systems review in Graham King's style
- `visual-review` — interactive HTML review dashboard for a PR or diff
- `pr-monitor` — a PR's CI status, why full CI has or has not triggered, and failure root cause against `main`
- `dynamo-issue-first` — start work from a tracked issue and reference it from the PR
- `repo-codeowners` — who reviews a change, fixing the `codeowners` check, changing review routing
- `dynamo-release-cherry-pick` — land a fix on a release branch: cherry-pick, release-only fix, or external Cherry-Pick Request

**Design proposals (DEPs):**

- `dep-create` — propose a feature or architecture change as a Dynamo Enhancement Proposal issue
- `dep-status` — check where a DEP stands, or list DEPs by lifecycle state or area
- `dep-update` — move a DEP through triage, PIC assignment, review, and approval

**Docs** (any change under `docs/`, `examples/`, or `recipes/`):

- `dynamo-docs` — add, update, move, or remove docs-site content per the style guide
- `fern-components` — pick and use Fern MDX components on a page
- `fern-navigation` — Fern site navigation and structure in `docs.yml`

**Deploy and operate:**

- `find-serving-recipe` — find the best known serving configuration for a model, hardware, and workload, with provenance
- `dynamo-router-starter` — start or patch router modes and smoke-test the endpoint
- `dynamo-interconnect-check` — validate NIXL/UCX/NCCL readiness for disaggregated serving
- `troubleshoot-dynamo` — diagnose failed or unhealthy deployments

**Recipe optimization loop** — start from
[`agent-docs/guides/optimization/optimize-loop.md`](agent-docs/guides/optimization/optimize-loop.md),
which sequences these skills:

- `synthesize-user-workload` — interview the user, capture their baseline DGD, and write the workload contract
- `author-baseline-dgd` — draft a baseline DGD when no recipe matches, for the user's confirmation
- `deploy-dynamo-recipe` — deploy one assigned DGD and smoke-test its API
- `configure-aiperf-benchmark` — freeze a comparable AIPerf workload for a deployed candidate
- `run-aiperf-benchmark` — run and collect one AIPerf Kubernetes benchmark
- `analyze-aiperf-results` — validate AIPerf results and compare them against SLOs and prior runs
- `consult-perf-knowledge` — select one evidence-backed optimization proposal
- `create-optimization-hypothesis` — turn a proposal into a challenger-ready DGD draft
- `perform-adversarial-review` — challenge a DGD draft before it consumes GPU time
- `report-skillpack-issue` — file a sanitized, operator-approved issue for a defect in these skills or guides

## Build and Test

### Published builds

Use a published build when you do not need to change it. Nightly CI builds `main` and
publishes the artifacts below; see
[Nightly Releases](docs/fern/pages/reference/general/nightly-releases.md) for details.

| Artifact | Where |
|----------|-------|
| Runtime images | `nvcr.io/nvidia/ai-dynamo/{vllm,sglang,tensorrtllm}-runtime-nightly`, tag `latest` or `YYYYMMDD-<shortsha>`; add `-efa` to either tag for the AWS EFA build |
| Component images | `nvcr.io/nvidia/ai-dynamo/{kubernetes-operator,dynamo-frontend,dynamo-planner}-nightly` |
| Python wheels | `ai-dynamo` and `ai-dynamo-runtime` (Python 3.11+) on `https://pypi.nvidia.com/`: `uv pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo` |
| Helm chart | `dynamo-platform-nightly` on NGC |
| Stable and prerelease builds, Rust crates | [Release Artifacts](docs/fern/pages/reference/general/release-artifacts.mdx) |

Pin the dated tag to reproduce a result; `latest` moves every night. To see the engine
version in an image, run `docker run --rm <image> pip show vllm`.

### Build from source

Install the system packages, Rust, and `uv` from
[Building from Source](docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md),
plus:

- protoc 3.15 or newer (CI uses 30.2 from the
  [protobuf releases](https://github.com/protocolbuffers/protobuf/releases))
- Python 3.11 or newer (`uv venv` can download one)

Python dev build (bindings + wheel, editable):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
cd lib/bindings/python
maturin develop --uv --features kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,request-trace-s3
cd -
uv pip install -e lib/gpu_memory_service
uv pip install -e .
python3 -m dynamo.frontend --help   # verify
```

The features match the CI wheel. Without `kv-indexer`, router tests that start a standalone
indexer fail. Python edits under `components/src/` take effect without a rebuild. After a
Rust change that Python uses, rerun the `maturin develop` command.

To work against an engine, build a `local-dev` image and mount your worktree into it.
[`container/README.md`](container/README.md) describes every target.

```bash
container/render.py --framework vllm --target local-dev --output-short-filename
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
  -t dynamo:latest-vllm-local-dev -f container/rendered.Dockerfile .
container/run.sh --mount-workspace -it --image dynamo:latest-vllm-local-dev
```

### Rust: build and test efficiently

Times on a 24-core workstation (October 2026, rustc 1.96.1), for relative cost:

| After you edit | Command | Time |
|----------------|---------|------|
| `dynamo-kv-router` | `cargo check -p dynamo-kv-router` | 2 s |
| `dynamo-kv-router` | `cargo test -p dynamo-kv-router --lib scheduling::` (277 tests) | 11 s |
| a sidecar | `cargo build` with the sidecar `-p` set below | 10–15 s |
| `dynamo-llm` | `cargo check -p dynamo-llm` | 17 s |
| `dynamo-llm` | `cargo test -p dynamo-llm --test http-service`, build and run | 39 s |
| `dynamo-llm` | `cargo test -p dynamo-llm --lib`: build, then run 3,731 tests | 70 s + 22 s |
| `dynamo-llm` | `cargo test -p dynamo-llm`: build all 44 test targets, before running any | 97 s |
| `dynamo-llm` | `maturin develop --uv` in `lib/bindings/python` | 94 s |
| nothing, in a new worktree | first `dynamo-llm` test build / sidecar set / bindings | 4 / 5.5 / 5.6 min |

The edit loop:

```bash
# 1. Type-check only the crate you are editing, until it compiles.
cargo check -p dynamo-kv-router

# 2. Run only the tests you need: name one target, then filter by test path.
cargo test -p dynamo-kv-router --lib scheduling::
cargo test -p dynamo-llm --test http-service
cargo test -p dynamo-llm --lib -- <name> <other_name>   # several filters at once

# 3. Before you push, format and lint the crates you touched as CI does.
cargo fmt --all
cargo clippy --locked --no-deps -p dynamo-kv-router --all-targets -- -D warnings
```

- **Name the test target** (`--lib` or `--test <name>`). Plain `cargo test -p dynamo-llm`
  builds all 44 test binaries.
- **Don't set `RUSTFLAGS`.** It replaces the flags in [`.cargo/config.toml`](.cargo/config.toml)
  and rebuilds every dependency, and unsetting it rebuilds them again.
- **Build the sidecars with one `-p` set:**
  `-p dynamo-vllm-sidecar -p dynamo-sglang-sidecar -p dynamo-trtllm-sidecar -p dynamo-lpx-sidecar -p dynamo-sidecar-testkit`.
  A different set changes features and rebuilds `dynamo-llm`.
- **Reuse your worktree.** A new one starts with an empty build cache (last row of the table).
- **Profiling:** `--profile profiling` is release with symbols; `-p dynamo-llm --features nvtx`
  adds NVTX ranges.
- CI's non-default feature checks are in [`pre-merge.yml`](.github/workflows/pre-merge.yml).
  Each is a separate cold build (about 4 min), so run only the one you touched.

On macOS, build and test `dynamo-llm` with `--no-default-features` unless you need `block-manager`.

### Python tests

Run the tests that cover your change, by file or test ID; leave marker sweeps to CI.
Use `python3 -m pytest` so the venv's pytest and packages are used.

```bash
# One test while you iterate, then the related tests in its file.
python3 -m pytest "tests/router/test_router_e2e_with_mockers.py::test_router_decisions[local_indexer-tcp]" -q
python3 -m pytest tests/router/test_router_e2e_with_mockers.py -k "test_router_decisions and not disagg" -q

# GPU end to end: the same, on the backend's serve tests.
python3 -m pytest tests/serve/test_vllm.py -k guided_decoding -s
```

`dynamo.mocker` stands in for an engine, so router, planner, and frontend changes can be
tested end to end without a GPU. Outside the dev containers, install the test plugins with
`uv pip install -r container/deps/requirements.test.txt`, and put `etcd` and `nats-server` on
`PATH`: integration fixtures start their own instances. Tests download their Hugging Face
model unless you pass `--models-dir`. Before writing tests, read
[`tests/README.md`](tests/README.md), [`.ai/pytest-guidelines.md`](.ai/pytest-guidelines.md),
and [`.ai/test-model-size-guardrails.md`](.ai/test-model-size-guardrails.md).

### Lint

Run `pre-commit install` once; the hooks, including the DCO sign-off check, then run on every
commit. To check before committing, run `pre-commit run --files <changed files>`.

## Pull Requests

Run the `pr-checklist` skill when you make or review a pull request.
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers DCO sign-off, commit titles, and how full CI is
triggered.
