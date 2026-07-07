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
- `dep-create` / `dep-status` / `dep-update` — create, track, and advance DEPs
- `dynamo-clone-hotpath-audit` — audit Rust hot-path `.clone()` calls
- `dynamo-docs` — Fern docs-site content per the style guide
- `dynamo-frontend-benchmark` — benchmark/profile the frontend against mock workers
- `graham-code-review` — strict Rust/systems review in Graham King's style
- `pr-monitor` — CI health check, failure root-cause, and skip analysis

**For deploying and operating Dynamo:**

- `dynamo-recipe-runner` — select, patch, and deploy Kubernetes recipes
- `dynamo-router-starter` — start/patch router modes with smoke checks
- `dynamo-interconnect-check` — validate NIXL/UCX/NCCL readiness for disaggregation
- `dynamo-troubleshoot` — diagnose failed or unhealthy deployments

**Adding a skill:** the folder name must equal the frontmatter `name` (kebab-case); the
`description` is third person and states what the skill does and when to use it; include
`license: Apache-2.0` and a `metadata:` block with `author` and `tags`. Changes under
`.agents/skills/` are validated by NVSkills CI — a maintainer comments `/nvskills-ci` on
the PR.

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
| `docs/`, `fern/` | Documentation sources and the Fern docs-site config — read [`docs/AGENTS.md`](docs/AGENTS.md) before editing |
| `examples/`, `recipes/` | Runnable examples and deployment recipes — also covered by [`docs/AGENTS.md`](docs/AGENTS.md) |
| `benchmarks/`, `tests/` | Benchmark harnesses and the top-level pytest suite |
| `.ai/` | Agent topic guidelines: `bash-launch-guidelines.md`, `ci-guidelines.md`, `linear-ticket-refs.md`, `pytest-guidelines.md`, `python-guidelines.md`, `test-model-size-guardrails.md` |
| `.agents/skills/` | Agent skills (see [Skills](#skills)) |

## Build

System prerequisites (Rust toolchain, `uv`, system libraries) and the VS Code / Cursor
devcontainer are covered in [`docs/contribution-guide.md`](docs/contribution-guide.md).

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
- Full CI on a PR runs only after a maintainer comments `/ok to test <sha>` with the short
  SHA of the latest commit; copy-pr-bot then creates the `pull-request/N` branch that
  triggers it. Fix failures before requesting human review.
- Architecture changes require a Dynamo Enhancement Proposal (DEP), filed as a GitHub
  issue on `ai-dynamo/dynamo` with `dep:*` labels (the `dep-create` skill automates this).

See [`docs/contribution-guide.md`](docs/contribution-guide.md) for the full workflow
(issue sizing, CODEOWNERS, review process).

## Docs, Examples, Recipes

Any change under `docs/`, `examples/`, or `recipes/` must follow
[`docs/AGENTS.md`](docs/AGENTS.md) and the
[documentation style guide](docs/documentation-style-guide.md): SPDX headers, Fern
frontmatter (no body `# H1`), GitHub-style admonitions, and backend casing
(vLLM / SGLang / TensorRT-LLM). The deterministic subset is enforced pre-merge.

## Cursor Cloud specific instructions

The Cursor Cloud VM has **no GPU** and **restricted network egress** (Hugging Face and
some other hosts are unreachable). The standard Build / Test / Lint commands above work;
the notes below cover only the non-obvious caveats for this environment. The startup
update script already activates `.venv` and refreshes the editable installs
(`maturin develop`, `-e .`, `-e lib/gpu_memory_service`, `requirements.test.txt`), so a
fresh session just needs `source .venv/bin/activate` before running anything.

- **Toolchain:** `cc`/`c++` default to `clang`, which cannot compile/link the vendored
  ZeroMQ (`zmq-sys`) C++ dependency (missing `libstdc++`). The snapshot pins `cc`→`gcc`
  and `c++`→`g++` via `update-alternatives`; keep it that way or `maturin develop` /
  `cargo build` will fail at the `zmq-sys` link step.
- **No GPU:** the real backends (`dynamo.vllm`, `dynamo.sglang`, `dynamo.trtllm`) and the
  `kvbm` bindings (need CUDA/NIXL) can't run here. Use `python -m dynamo.mocker` as a
  GPU-free worker. `pytest -m unit tests/` passes except the optional `kvbm` import checks
  in `tests/dependencies/test_kvbm_imports.py` (kvbm wheel not built) — expected.
- **Offline models:** never rely on downloading from Hugging Face. Use the local
  fixture models under `lib/llm/tests/data/sample-models/` and set `HF_HUB_OFFLINE=1`
  `TRANSFORMERS_OFFLINE=1`. For `/v1/chat/completions` the model dir must contain a
  `chat_template` (e.g. `mock-llama-3.1-8b-instruct`; `TinyLlama_v1.1` has none and only
  works for non-chat paths).
- **Run without etcd/NATS:** the frontend defaults to etcd discovery, which is not running.
  Pass `--discovery-backend file` to both the frontend and the worker and point them at the
  same `DYN_FILE_KV` dir (defaults to `$TMPDIR/dynamo_store_kv`). The default `tcp` request
  plane and `zmq` event plane need no external services.
- **GPU-free smoke test** (frontend + KV router + mock worker), each in its own shell:
  ```bash
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DYN_FILE_KV=/tmp/dynamo_store_kv
  python -m dynamo.frontend --http-port 8000 --router-mode round-robin --discovery-backend file
  python -m dynamo.mocker --model-path lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct \
      --discovery-backend file --speedup-ratio 1000
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
  ```
  The mocker simulates scheduling/KV/timing and returns token counts (`content` is `null`
  because it emits placeholder token IDs, not real text) — that is expected.
- **`pre-commit run --all-files`:** the deterministic hooks (isort, black, flake8,
  clang-format, codespell, ruff, file checks) pass. The `pytest-marker-report` hook needs
  the full backend deps installed to collect every test, so it reports gaps on this
  GPU-free VM; treat its result as informational here.
