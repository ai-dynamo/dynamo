<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo — Agent Guide

[`README.md`](README.md) explains what Dynamo is and what it supports. This guide covers
working in the repo: where code lives, which skill fits a task, and how to build and test
a change.

## Repository Map

| Path | Contents |
|------|----------|
| `lib/` | Rust workspace crates: `runtime`, `llm`, `kv-router`, `kvbm-*`, `mocker`, `sidecar/*` (engine gRPC sidecars), and more (see the root [`Cargo.toml`](Cargo.toml) `[workspace] members`), plus `bindings/python` — the PyO3 extension crate, built via maturin and deliberately excluded from the workspace |
| `components/src/dynamo/` | Python packages: `frontend`, `planner`, `router`, `vllm`/`sglang`/`trtllm`/`tokenspeed` backends, `mocker`, `profiler`, and more |
| `deploy/` | Kubernetes `operator`, Helm charts, `inference-gateway` ext-proc, `observability` |
| `container/` | Dockerfiles and build scripts for runtime and dev images |
| `docs/fern/` | Fern docs site: `pages/` holds every page, the rest is site config (`index.yml`, `docs.yml`, `main.css`, `components/`, `scripts/`, `translations/`) |
| `examples/`, `recipes/` | Runnable examples and deployment recipes |
| `benchmarks/`, `tests/` | Benchmark harnesses and the top-level pytest suite |
| `.ai/` | Agent topic guidelines: `ais-perf-model-guidelines.md` (canonical AISimulate interface), `bash-launch-guidelines.md`, `ci-guidelines.md`, `linear-ticket-refs.md`, `pytest-guidelines.md`, `python-guidelines.md`, `test-model-size-guardrails.md` |
| `.agents/skills/` | Agent skills (see [Skills](#skills)) |
| `agents/`, `agent-docs/` | Role contracts, guides, and rules for the recipe optimization loop |

## Ecosystem

Repositories this repo integrates with:

| Repo | Role |
|------|------|
| [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [TokenSpeed](https://github.com/lightseekorg/tokenspeed) | Inference engines that Dynamo serves |
| [NIXL](https://github.com/ai-dynamo/nixl) | High-throughput inference data-transfer library (KV-cache transfer over RDMA/NVLink) that underpins disaggregated serving |
| [AIPerf](https://github.com/ai-dynamo/aiperf) | Benchmarking and load-generation tool used by the benchmarking guides |
| [AISimulate](https://pypi.org/project/aisimulate/) | Predicts serving behavior and searches deployment configurations offline without requiring a GPU cluster |
| [ModelExpress](https://github.com/ai-dynamo/modelexpress) | Streams model weights GPU-to-GPU via NIXL for fast replica cold-start |
| [Grove](https://github.com/ai-dynamo/grove) | Kubernetes operator for topology-aware gang scheduling |

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

**Adding a skill:** the folder name must equal the frontmatter `name` (kebab-case); the
`description` is third person, states what the skill does and when to use it, and is at
most 1024 characters; include `license: Apache-2.0` and a `metadata:` block with `author`
and `tags`. List the skill in this section — the index must match `.agents/skills/`
exactly. All of this is enforced by `scripts/validate_skills.py` (pre-commit hook
`validate-skills`). Changes under `.agents/skills/` are also validated by NVSkills CI —
a maintainer comments `/nvskills-ci` on the PR.

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

- **protoc 3.15 or newer.** Ubuntu 22.04's `protobuf-compiler` is 3.12, and the `dynamo-llm`
  build fails on `relay.proto`. CI installs protoc 30.2 from the
  [protobuf releases](https://github.com/protocolbuffers/protobuf/releases).
- **Python 3.11 or newer.** `uv venv` finds or downloads one; Ubuntu 22.04's `python3` is 3.10.

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

Most Rust builds wait on one serial chain of crates that ends in `dynamo-llm`. More
cores, sccache, or a remote cache barely help; the cost is set by which crates you build
and which test binaries you link. Measured on a 24-core workstation (October 2026,
rustc 1.96.1), for relative cost:

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

- **Keep one `-p` set per area.** Cargo resolves features across the packages you name,
  so each `-p` set gets its own build of the shared crates; the first build of a new set
  costs a rebuild. For the sidecars, always use
  `-p dynamo-vllm-sidecar -p dynamo-sglang-sidecar -p dynamo-trtllm-sidecar -p dynamo-lpx-sidecar -p dynamo-sidecar-testkit`.
- **Name the test target.** `cargo test -p dynamo-llm` with no target builds the unit-test
  harness (a second compile of `dynamo-llm`), all 44 test binaries, and the doc tests.
  Use `--lib` or `--test <name>`.
- **Keep build flags stable.** [`.cargo/config.toml`](.cargo/config.toml) sets the
  `rustflags`. A `RUSTFLAGS` variable replaces them and recompiles every dependency, and
  so does changing it back; a change to the dev profile or the toolchain does the same.
  Set `CARGO_PROFILE_DEV_DEBUG=line-tables-only` once and leave it set: a cold
  `dynamo-llm` test build took 4 min with it and 5.5 min with full debug info.
- **Reuse your worktree.** A new worktree starts with an empty build cache; switching
  branches in an existing one rebuilds only the crates that changed.
- **Measure performance on optimized builds.** Debug timings mean nothing. Use
  `--release` (`maturin develop --uv --release` for the bindings). To profile with perf or
  Nsight, build with `--profile profiling`, which is release with symbols;
  `-p dynamo-llm --features nvtx` adds NVTX ranges.
- **Run only the feature lanes you touch.** CI also checks some crates with non-default
  features, such as `dynamo-llm --no-default-features` and
  `dynamo-kv-router --features standalone-indexer`; the exact commands are in
  [`.github/workflows/pre-merge.yml`](.github/workflows/pre-merge.yml). Each feature set is
  a separate build: `dynamo-llm --no-default-features` takes 4 min the first time.

On macOS, run `dynamo-llm` checks and targeted tests with `--no-default-features` unless the
target explicitly requires `block-manager`. The default feature enables Linux/CUDA-oriented NIXL,
NUMA, and `O_DIRECT` code that is not a valid general-purpose macOS validation path.

### Python tests

Run the tests that cover your change, by file or test ID; leave marker sweeps to CI.
Use `python3 -m pytest` so the venv's pytest and packages are used.

```bash
# One test while you iterate, then the related tests in its file.
python3 -m pytest "tests/router/test_router_e2e_with_mockers.py::test_router_decisions[local_indexer-tcp]" -q
python3 -m pytest tests/router/test_router_e2e_with_mockers.py -k "test_router_decisions and not disagg" -q

# GPU end to end: the same, on the backend's serve tests.
python3 -m pytest tests/serve/test_vllm.py -k guided_decoding -s

# Before you push, list the pre-merge CPU tests in the area you changed.
python3 -m pytest --collect-only -q -m "pre_merge and gpu_0" tests/router
```

`dynamo.mocker` stands in for an engine, so router, planner, and frontend changes can be
tested end to end without a GPU. Outside the dev containers, install the test plugins with
`uv pip install -r container/deps/requirements.test.txt`, and put `etcd` and `nats-server` on
`PATH`: integration fixtures start their own instances. Tests download their Hugging Face
model unless you pass `--models-dir`. Markers are strict (`--strict-markers`); the full marker
list lives in [`pyproject.toml`](pyproject.toml) `[tool.pytest.ini_options]`, including GPU
gating (`gpu_0` … `gpu_8`). Read [`tests/README.md`](tests/README.md),
[`.ai/pytest-guidelines.md`](.ai/pytest-guidelines.md), and
[`.ai/test-model-size-guardrails.md`](.ai/test-model-size-guardrails.md) before writing
tests.

### Lint

```bash
pre-commit run --files <changed files>   # hooks on what you changed
pre-commit run --all-files               # every hook; run `pre-commit install` first (it also installs the DCO commit-msg hook)
```

## Pull Requests

Run the `pr-checklist` skill when you make or review a pull request.
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers DCO sign-off, commit titles, and how full CI is
triggered.
