# Dynamo Testing Guidelines

This document provides instructions for organizing, marking, and running tests in the Dynamo project. Follow these guidelines to ensure consistency and maintainability across the test suite.

Dynamo has three areas of tests and checks:

1. **[Rust Testing](#rust-testing)** -- Covers the Rust crates under `lib/`. Has unit and integration tests. CI also enforces format, lint, and license checks before merge.
2. **[Python Testing (pytest)](#python-testing-pytest)** -- Covers Python components and cross-component workflows. Has unit, integration, and E2E tests. Uses pytest markers to select tests by lifecycle stage, hardware, and framework.
3. **Miscellaneous checks** -- Format (`cargo fmt`, `ruff`), lint (`clippy`, `pre-commit`), license (`cargo-deny`), unused dependencies (`cargo machete`), doc build (`cargo doc`). These run as part of CI and are documented in [Running Rust Checks and Tests](#running-rust-checks-and-tests).

All tests run inside containers. See the [Container Development Guide](../container/README.md) for how to build and launch one.

Each area can have one or more of the following types of tests:

1. **Unit** -- Exercises a single function, class, or module in isolation. No external services, no GPU. Each test typically runs in milliseconds; all unit tests combined may take <5 minutes.
2. **Integration** -- Wires multiple components together using **mock engines** (`dynamo.mocker`) and **real infrastructure** (ETCD for service discovery, NATS for messaging, if enabled). Validates that the router, planner, frontend gRPC, and similar subsystems work together without launching a real inference engine. No GPU required. Each test typically runs in seconds; all integration tests combined may take <30 minutes.
3. **End-to-End (E2E)** -- Starts a **real inference engine** (vLLM, SGLang, or TRT-LLM), sends requests through the frontend, and validates responses. Requires GPU. Each test typically runs in minutes; the full E2E suite may take several hours.

It is absolutely important to be mindful of how long a test you write takes. Slow tests have a compounding cost: they burn GPU-hours in CI (GPUs are expensive and shared), they discourage engineers from running suites locally (so bugs slip through to CI), and they slow down the entire team's development velocity. A test suite that takes too long becomes a test suite that nobody runs. When adding or modifying tests, include a per-test time estimate in your PR description -- CI GPU resources are limited and these estimates help the team schedule tests across pre-merge, nightly, and weekly pipelines.

Timings in this document are approximate, measured on a 32-core machine as of Q1 2026. They will vary with hardware and codebase size.

---

## Test Organization: Where to Store Tests

### Directory Structure
```
dynamo/
├── lib/
│   ├── runtime/
│   │   ├── src/
│   │   │   └── lib.rs              # Rust code + unit tests inside
│   │   └── tests/                  # Rust integration tests for runtime
│   ├── llm/
│   │   ├── src/
│   │   │   └── lib.rs              # Rust code + unit tests inside
│   │   └── tests/                  # Rust integration tests for llm
│   └── ...
├── components/
│   └── src/dynamo/
│       ├── vllm/
│       │   └── tests/              # Python unit/integration tests for vllm backend
│       ├── trtllm/
│       │   └── tests/              # Python unit/integration tests for trtllm backend
│       ├── sglang/
│       │   └── tests/              # Python unit/integration tests for sglang backend
│       ├── fastvideo/
│       │   └── tests/              # Python unit/integration tests for fastvideo backend
│       ├── common/
│       │   └── tests/              # Python unit/integration tests for common utils
│       ├── planner/
│       ├── router/
│       ├── frontend/
│       ├── profiler/
│       └── ...
├── tests/                          # End-to-end and cross-component tests
│   ├── serve/                      # Serve E2E tests (vllm, sglang, trtllm, fastvideo)
│   ├── kvbm_integration/           # KVBM integration tests
│   ├── gpu_memory_service/         # GPU Memory Service E2E tests
│   ├── fault_tolerance/            # Fault tolerance, migration, cancellation
│   ├── deploy/                     # Deployment tests
│   ├── frontend/                   # Frontend HTTP/gRPC tests
│   ├── router/                     # Router E2E tests
│   ├── mm_router/                  # Multimodal router tests
│   ├── lmcache/                    # LM cache tests
│   ├── basic/                      # Basic backend tests
│   └── utils/                      # Shared test utilities
├── benchmarks/                     # Performance/load benchmarks
│   ├── router/
│   ├── llm/
│   └── ...
```
- Place **unit/integration tests** for a component in its `tests/` subfolder under `components/src/dynamo/<component>/tests/`.
- Place **end-to-end (E2E) tests** and cross-component tests in `tests/`.
- Name test files as `test_<component>_<flow>.py` for clarity.

### Test Types and Locations

**Rust tests** (`cargo test`) -- each test typically takes 100 ms to 30 s:

| Type              | Description                              | Location                                     |
|-------------------|------------------------------------------|----------------------------------------------|
| Unit              | Single function/class, inline tests      | `lib/<crate>/src/` (`#[cfg(test)]` modules)  |
| Integration       | Cross-module, feature-gated              | `lib/<crate>/tests/`                         |

**Python tests** (`pytest`):

| Type               | Description                           | Location                                      |
|--------------------|---------------------------------------|-----------------------------------------------|
| Unit               | Single function/class, isolated       | `components/src/dynamo/<component>/tests/`    |
| Integration        | Interactions between modules/services | `components/src/dynamo/<component>/tests/`    |
| End-to-End         | User workflows, CLI, API              | `tests/serve/`, `tests/deploy/`, etc.         |
| KVBM Integration   | KV block manager integration          | `tests/kvbm_integration/`                     |
| GPU Memory Service | GPU Memory Service E2E                | `tests/gpu_memory_service/`                   |
| Router             | Router E2E with backends              | `tests/router/`                               |
| Planner            | Planner unit + integration tests      | `components/src/dynamo/planner/tests/`        |
| Frontend           | Frontend HTTP/gRPC tests              | `tests/frontend/`                             |
| Profiler           | Profiler unit + integration tests     | `components/src/dynamo/profiler/tests/`       |
| Global Planner     | Global planner unit tests             | `components/src/dynamo/global_planner/tests/` |
| Fault Tolerance    | Chaos, migration, cancellation        | `tests/fault_tolerance/`                      |
| Deployment         | Deployment validation                 | `tests/deploy/`                               |
| Benchmark          | Performance/load                      | `benchmarks/`                                 |

---

## Test Marking: How to Mark Tests

Markers are required for all tests. They are used for test selection in CI and local runs.

### Marker Requirements
- Every test must have at least one **Lifecycle** marker, and **Test Type** and **Hardware** markers.
- **Component/Framework** markers are required as applicable.

### Marker Table
| Category                | Marker(s)                                                        | Description                        |
|-------------------------|------------------------------------------------------------------|------------------------------------|
| Lifecycle [required]    | pre_merge, post_merge, nightly, weekly, release                  | When the test should run           |
| Test Type [required]    | unit, integration, e2e, benchmark, performance, stress, multimodal | Nature of the test               |
| Hardware [required]     | gpu_0, gpu_1, gpu_2, gpu_4, gpu_8, h100                         | Number/type of GPUs required       |
| VRAM (profiled)         | profiled_vram_gib(N)                                                         | Actual peak VRAM observed by nvidia-smi during profiling (includes CUDA overhead). Used for `--max-vram-gib=N` filtering and GPU-parallel scheduler budget tracking. |
| vLLM KV cache bytes     | requested_vllm_kv_cache_bytes(N)                                             | (vLLM only) Exact KV cache bytes. Sets `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` → `--kv-cache-memory-bytes`. Deterministic, parallel-safe. |
| SGLang KV tokens        | requested_sglang_kv_tokens(N)                                                          | (SGLang only) Max KV cache tokens. Sets `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` → `--max-total-tokens`. Deterministic, parallel-safe. |
| TRT-LLM KV tokens      | requested_trtllm_kv_tokens(N)                                                          | (TRT-LLM only) Max KV cache tokens. Sets `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` → `KvCacheConfig.max_tokens` via `--override-engine-args`. Deterministic, parallel-safe. |
| TRT-LLM VRAM GiB       | requested_trtllm_vram_gib(N)                                                           | (TRT-LLM only) Max VRAM in GiB. Sets `_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES` → `KvCacheConfig.max_gpu_total_bytes` via `--override-engine-args`. For non-text workloads (video/image diffusion) where token-based control doesn't apply. |
| Component/Framework     | vllm, trtllm, sglang, fastvideo, kvbm, kvbm_concurrency, planner, router   | Backend or component specificity   |
| Infrastructure          | k8s, deploy, fault_tolerance                                     | Infrastructure/environment needs   |
| Execution               | parallel                                                         | Test can run in parallel with pytest-xdist. Must use dynamic port allocation (`alloc_ports`) and not share resources (e.g. filesystem) |
| Other                   | slow, skip, xfail, custom_build, model, aiconfigurator           | Special handling                   |

### Example (vLLM)
```python
@pytest.mark.pre_merge
@pytest.mark.integration
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(20.5)  # actual nvidia-smi peak
@pytest.mark.requested_vllm_kv_cache_bytes(942_054_000)  # KV cache cap (2x safety over min=471_027_000)
@pytest.mark.vllm
def test_kv_cache_behavior():
    ...
```

### Example (SGLang with token cap)
```python
@pytest.mark.pre_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(3.7)   # actual nvidia-smi peak at recommended token count
@pytest.mark.requested_sglang_kv_tokens(96)     # KV cache cap (2x safety over min=48)
@pytest.mark.timeout(265)
@pytest.mark.sglang
def test_sglang_aggregated():
    ...
```

### Example (TRT-LLM with token cap)
```python
@pytest.mark.pre_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(3.9)   # actual nvidia-smi peak at recommended token count
@pytest.mark.requested_trtllm_kv_tokens(2592)   # KV cache cap (2x safety over min=1296)
@pytest.mark.timeout(300)
@pytest.mark.trtllm
def test_trtllm_aggregated():
    ...
```

### Example (TRT-LLM diffusion — no KV cache)
```python
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.trtllm
# Diffusion models don't use KV cache, so requested_trtllm_kv_tokens doesn't apply
# and requested_trtllm_vram_gib (KvCacheConfig.max_gpu_total_bytes) has no effect —
# the VRAM is model weights + activations. Only profiled_vram_gib is meaningful.
@pytest.mark.profiled_vram_gib(17.1)  # actual nvidia-smi peak
@pytest.mark.timeout(600)
def test_trtllm_video_diffusion():
    ...
```

### VRAM Markers and Filtering

Markers differ by engine:

**vLLM** uses byte-based KV cache control:
- **`profiled_vram_gib(N)`** — actual peak from nvidia-smi. Used for `--max-vram-gib` filtering and scheduler budget.
- **`requested_vllm_kv_cache_bytes(N)`** — exact KV cache bytes. Sets `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` → `--kv-cache-memory-bytes`. Deterministic and parallel-safe.

**SGLang** uses token-based control:
- **`profiled_vram_gib(N)`** — actual peak from nvidia-smi at the recommended token count. Used for `--max-vram-gib` filtering and scheduler budget.
- **`requested_sglang_kv_tokens(N)`** — max KV cache tokens. Sets `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` → `--max-total-tokens`. SGLang's default `--mem-fraction-static` is never overridden; the token cap is the sole allocation control. Deterministic and parallel-safe (see `examples/common/gpu_utils.md`).

**TRT-LLM** uses token-based control (text models) or byte-based control (diffusion models):
- **`profiled_vram_gib(N)`** — actual peak from nvidia-smi. Used for `--max-vram-gib` filtering and scheduler budget.
- **`requested_trtllm_kv_tokens(N)`** — max KV cache tokens for text models. Sets `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` → `KvCacheConfig.max_tokens` via `--override-engine-args` JSON. Deterministic and parallel-safe.
- **`requested_trtllm_vram_gib(N)`** — max VRAM in GiB for non-text workloads (video/image diffusion). Sets `_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES` → `KvCacheConfig.max_gpu_total_bytes` via `--override-engine-args` JSON. Note: diffusion models don't use KV cache, so this parameter may have no effect — `profiled_vram_gib` alone is sufficient for scheduler budget tracking.
- TRT-LLM requires JSON merging for `--override-engine-args`, handled by `build_trtllm_override_args_with_mem` in `gpu_utils.sh` (separate from `build_vllm_gpu_mem_args` / `build_sglang_gpu_mem_args`).

`--max-vram-gib=N` deselects tests whose `profiled_vram_gib` exceeds N. Tests without a VRAM marker are also deselected (unknown VRAM = unsafe for parallel). To add a test to the pool, profile it with `tests/utils/profile_pytest.py` (see [GPU VRAM Profiler](#gpu-vram-profiler-profile_pytestpy)).

### GPU-Parallel Execution

GPU tests run concurrently via a custom VRAM-aware scheduler (`tests/utils/pytest_parallel_gpu.py`). This is separate from `pytest-xdist` because:

1. **VRAM budget**: xdist has no GPU memory awareness — two 20 GiB tests on a 48 GiB GPU will OOM.
2. **Profiling race**: engines snapshot free memory during init; concurrent startups corrupt each other. The scheduler staggers launches (VRAM stability check) and retries transient failures.
3. **Engine-specific allocation**: each test gets a constrained allocation so it uses only its budgeted share. xdist has no mechanism for this.
   - **vLLM**: `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES = N` → `--kv-cache-memory-bytes` (from `requested_vllm_kv_cache_bytes` marker). Byte-based cap is deterministic and doesn't depend on current free memory, making it inherently parallel-safe. Uses `build_vllm_gpu_mem_args` in `gpu_utils.sh`.
   - **SGLang**: `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS = N` → `--max-total-tokens` (from `requested_sglang_kv_tokens` marker). Token-based cap is deterministic and doesn't depend on current free memory, making it inherently parallel-safe. Uses `build_sglang_gpu_mem_args` in `gpu_utils.sh`.
   - **TRT-LLM**: `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS = N` → `KvCacheConfig.max_tokens` via `--override-engine-args` JSON (from `requested_trtllm_kv_tokens` marker). Token-based cap is deterministic and parallel-safe. Uses `build_trtllm_override_args_with_mem` in `gpu_utils.sh` (separate function because TRT-LLM requires JSON merging).

```bash
# Dry-run: preview which tests fit and the GPU plan
python3 -m pytest --max-vram-gib=24 --dry-run -m "gpu_1 and vllm" tests/serve/test_vllm.py

# Run pre-merge vllm tests in parallel
python3 -m pytest --max-vram-gib=6 -n auto -m "gpu_1 and vllm and not nightly and not post_merge" tests/serve/test_vllm.py

# Run all (pre+post merge) with live output
python3 -m pytest --max-vram-gib=48 -n auto -sv -m "gpu_1 and vllm and not nightly" tests/serve/test_vllm.py tests/frontend/test_vllm.py

# SGLang tests
python3 -m pytest --max-vram-gib=48 -n auto -m "gpu_1 and sglang" tests/serve/test_sglang.py

# Tests that still need profiling
python3 -m pytest --dry-run -m "(gpu_1 or gpu_2) and not profiled_vram_gib" tests/serve/
```

Example output (6 SGLang tests, RTX 6000 Ada 48 GiB):
```
GPU parallel: 6 tests, 7 concurrent slots, GPU0 (48 GiB, 43 GiB multi-proc budget)

[w0] tests/serve/test_sglang.py::...completions_only-2]     profiled= 14.9 GiB  req_kv_tokens=  1024  timeout=420s
[w1] tests/serve/test_sglang.py::...multimodal_agg_qwen-2]  profiled= 20.2 GiB  req_kv_tokens=   512  timeout=280s
[w2] tests/serve/test_sglang.py::...aggregated-2]            profiled=  6.0 GiB  req_kv_tokens=  1024  timeout=240s
...

[w0] tests/serve/...completions_only-2] (GPU0, profiled 14.9 GiB, req_kv_tokens=  1024) RUNNING
[w1] tests/serve/...multimodal_agg_qwen-2] (GPU0, profiled 20.2 GiB, req_kv_tokens=   512) RUNNING
[elapsed 10s] GPU0: 0.6/48 GiB [w0(10s), w1(5s)] [queued: w2, w3, w4, w5]
[w1] tests/serve/...multimodal_agg_qwen-2] PASSED [31s]
[w0] tests/serve/...completions_only-2] PASSED [76s]
...
=============== 6 passed in 111.00s (1:51) (vs 228s seq, 2.1x) ===============
```

### Lifecycle Marker Note
Use the marker for the earliest pipeline stage where the test must run (e.g., `@pytest.mark.pre_merge`). This ensures the test is included in that stage and all subsequent ones (e.g., nightly, release), as CI pipelines select tests marked for earlier stages.

**Example:**
If a test is marked with `@pytest.mark.pre_merge`, and the nightly pipeline runs:
```bash
pytest -m "e2e and (pre_merge or post_merge or nightly)"
```
then this test will be included in the nightly run as well.

---

## Rust Testing

### Organization
- **Unit tests** are placed within the corresponding Rust source files (e.g., `lib.rs`) using `#[cfg(test)]` modules.
- **Integration tests** are placed in the crate's `tests/` directory and must be gated behind the `integration` feature.

### Running Rust Checks and Tests

Run these in order. Format and lint checks are fast; fix any issues before running tests.
These commands are derived from [`.github/workflows/pre-merge.yml`](../.github/workflows/pre-merge.yml).

```bash
# Format check (typically <5s)
cargo fmt -- --check

# Clippy lint (typically <5min first run, faster with cache)
cargo clippy --no-deps --all-targets -- -D warnings

# License check (typically <15s)
cargo-deny -L error --all-features check licenses bans --config deny.toml

# Unused dependency check (typically <15s)
cargo machete

# Compile tests without executing (typically <5min first run; catches build errors early)
cargo test --locked --no-run

# Doc tests (typically <5min)
cargo doc --no-deps && cargo test --locked --doc

# Unit tests -- most important for code correctness (typically <5min)
cargo test --locked --all-targets

# Integration tests (may require ETCD/NATS running; typically <10min)
cargo test --features integration
```


### Additional Options
- **Feature gates:** Use Cargo features to run specific test subsets, e.g. `cargo test --features planner`. Integration tests must be behind the `integration` feature gate.
- **Ignored tests:** Use `#[ignore]` to mark slow or special-case tests. Run them explicitly with `cargo test -- --ignored`.

### Example
```rust
#[cfg(test)]
mod kv_cache_tests {
    #[test]
    fn test_kv_cache_basic() {
        // ...
    }

    #[test]
    #[ignore]
    fn test_kv_cache_long_running() {
        // ...
    }
}
```

### CI Integration
- CI runs the commands listed in [Running Rust Checks and Tests](#running-rust-checks-and-tests) across 4 workspace directories: `.`, `lib/bindings/python`, `lib/runtime/examples`, `lib/bindings/kvbm`. See [`.github/workflows/pre-merge.yml`](../.github/workflows/pre-merge.yml) for the exact steps.

---

## Python Testing (pytest)

### Prerequisites

This section assumes you are already inside a running **runtime**, **local-dev**, or **dev** container. If not, see the [Container Development Guide](../container/README.md) to build and launch one. The typical workflow is:

1. Build a development container (`render.py ...` + `docker build ...`)
2. Launch it (`run.sh ...`)
3. Inside the container, compile code and run tests

All commands below are meant to be run **inside the container**.

**Local-dev / dev containers** -- you must compile the Rust bindings before running pytest. Without this step, tests that import `dynamo._internal` will fail with `ImportError`:
```bash
cargo build --locked --features dynamo-llm/block-manager --workspace
cd lib/bindings/python && maturin develop --uv && cd -
```

**Runtime containers** -- binaries are pre-built, no compilation needed. Just run pytest.

Sanity check (optional but recommended) -- verify the environment is wired up correctly:
```bash
deploy/sanity_check.py                        # local-dev / dev containers
deploy/sanity_check.py --runtime-check-only   # runtime containers
```

### Environment Setup
- Use the dev container for consistency.
- Install dependencies as specified in `pyproject.toml`.
- Set the `HF_TOKEN` environment variable for HuggingFace downloads:
  ```bash
  export HF_TOKEN=your_token_here
  ```
- Model cache is located at `~/.cache/huggingface` to avoid repeated downloads.

### Running Python Tests

Python has many markers and variations. Tests are tagged with **lifecycle** markers (`pre_merge`, `post_merge`, `nightly`) that control *when* they run in CI, and **test-type** markers (`unit`, `integration`, `e2e`) that describe *what* they test.

**Local development (quick feedback)** -- run these before submitting to CI:
```bash
# Unit tests -- fastest (typically <15s)
pytest -m "unit and pre_merge" -v --tb=short

# Integration tests -- uses mock engines with real infrastructure (ETCD, NATS); no GPU needed (typically <10min)
pytest -m "integration and pre_merge" -v --tb=short

# E2E smoke test -- launches a full inference engine, sends requests, validates responses (typically <5min)
# vllm
pytest tests/serve/test_vllm.py::test_serve_deployment[aggregated] -v --tb=short
# sglang
pytest tests/serve/test_sglang.py::test_sglang_deployment[aggregated-2] -v --tb=short
# trtllm
pytest tests/serve/test_trtllm.py::test_deployment[aggregated-2] -v --tb=short
# fastvideo
python3 -m pytest tests/serve/test_fastvideo.py::test_fastvideo_deployment[aggregated] -v --tb=short
```

**Pre-merge CI equivalent** -- this is what [`container-validation-dynamo.yml`](../.github/workflows/container-validation-dynamo.yml) runs on every PR. Tests marked `parallel` run with `pytest-xdist`; the rest run sequentially:
```bash
# Parallel pre-merge tests (4 workers, CPU-only; typically <5min)
python3 -m pytest -m "pre_merge and parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0" -n 4 --dist=loadscope -v --tb=short

# Sequential pre-merge tests (CPU-only; typically <10min)
python3 -m pytest -m "pre_merge and not parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0" -v --tb=short
```

> **Parallel vs sequential:** CPU-only tests (`gpu_0`) marked `parallel` run with `pytest-xdist` (`-n auto` or `-n <workers>`, `--dist=loadscope`). GPU tests (`gpu_1`, `gpu_2`, etc.) run sequentially by default, but can run in parallel with `--max-vram-gib=N -n auto` (uses a custom VRAM-aware scheduler, not xdist). See [`.github/actions/pytest/action.yml`](../.github/actions/pytest/action.yml).

**Full E2E suite** -- launches engines for every test configuration; slowest, requires GPU and a framework container (typically <30min depending on framework and model):
```bash
# Sequential (default)
pytest -m "vllm and e2e and gpu_1" -v --tb=short
pytest -m "sglang and e2e and gpu_1" -v --tb=short
pytest -m "trtllm and e2e and gpu_1" -v --tb=short
python3 -m pytest -m "fastvideo and e2e and gpu_1" -v --tb=short

# GPU-parallel (VRAM-aware scheduling, ~2x faster on 48 GiB GPU)
# Only tests with profiled_vram_gib markers are selected; -n auto calculates
# concurrent slots from GPU VRAM / smallest test. See "GPU-Parallel Execution" below.
python3 -m pytest --max-vram-gib=48 -n auto -m "gpu_1 and sglang" tests/serve/test_sglang.py -v
python3 -m pytest --max-vram-gib=48 -n auto -m "gpu_1 and vllm" tests/serve/test_vllm.py -v
```

**Post-merge equivalent** -- CI runs `(pre_merge or post_merge)` after merge, which adds slower tests on top of the pre_merge set. **Running the full post-merge suite locally can take several hours per framework** (model downloads, GPU inference, multi-GPU coordination). For day-to-day development, before you submit to CI, use the `pre_merge` commands above for quicker feedback. See [`.github/workflows/post-merge-ci.yml`](../.github/workflows/post-merge-ci.yml) for exact markers:
```bash
pytest -m "(pre_merge or post_merge) and vllm and gpu_0" -n auto --dist=loadscope -v --tb=short
pytest -m "(pre_merge or post_merge) and vllm and gpu_1" -v --tb=short
```

- Run by component:
  ```bash
  pytest -m planner
  pytest -m kvbm
  ```
- Show print/log output:
  ```bash
  pytest -s
  ```
- CI runs use similar instructions from inside a container. For example, running E2E tests as part of the post-merge suite:
  ```bash
  ./container/run.sh --image $VLLM_IMAGE_NAME --name $VLLM_CONTAINER_NAME -- pytest -m "(pre_merge or post_merge) and vllm and e2e and gpu_1"
  ```

### Running tests locally outside of a container

To run tests outside of the development container, ensure that you have properly set up your environment and have installed the following dependencies in your `venv`:

```bash
uv pip install pytest-mypy
uv pip install pytest-asyncio
```

---

## CI Pipeline Overview

It is highly recommended that you run tests thoroughly on your local machine before submitting to CI. Local iteration is faster, gives you immediate feedback, and avoids burning shared CI GPU resources on avoidable failures. The following stages are what CI runs -- you can (and should) run the same commands on your machine before submitting to CI.

Source workflow files (see [`.github/workflows/`](../.github/workflows/) for the full set):
- **Pre-merge (Rust):** [`.github/workflows/pre-merge.yml`](../.github/workflows/pre-merge.yml)
- **Pre-merge (Python):** [`.github/workflows/container-validation-dynamo.yml`](../.github/workflows/container-validation-dynamo.yml)
- **Post-merge:** [`.github/workflows/post-merge-ci.yml`](../.github/workflows/post-merge-ci.yml) -> [`.github/workflows/build-test-distribute-flavor.yml`](../.github/workflows/build-test-distribute-flavor.yml)
- **Nightly:** [`.github/workflows/nightly-ci.yml`](../.github/workflows/nightly-ci.yml)
- **Pytest action:** [`.github/actions/pytest/action.yml`](../.github/actions/pytest/action.yml)

### Pre-merge (every PR)

Two workflows run on every PR. See [`pre-merge.yml`](../.github/workflows/pre-merge.yml) and [`container-validation-dynamo.yml`](../.github/workflows/container-validation-dynamo.yml).

**Rust checks** (only if Rust files changed) -- runs `pre-commit`, then the full sequence from [Running Rust Checks and Tests](#running-rust-checks-and-tests) across 4 workspace dirs (`.`, `lib/bindings/python`, `lib/runtime/examples`, `lib/bindings/kvbm`): format, clippy, cargo-deny, machete, compile, doc tests, unit tests.

**Python tests** (framework-agnostic, CPU-only, inside a dynamo container):

| Stage | Marker expression | Local equivalent |
|-------|------------------|-----------------|
| Parallel (xdist, 4 workers) | `pre_merge and parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0` | `python3 -m pytest -m "pre_merge and parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0" -n 4 --dist=loadscope -v --tb=short` |
| Sequential | `pre_merge and not parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0` | `python3 -m pytest -m "pre_merge and not parallel and not (vllm or sglang or trtllm or fastvideo) and gpu_0" -v --tb=short` |

### Post-merge (push to release branches)

Runs per framework (vllm, sglang, trtllm, fastvideo). Each framework goes through: **Build** -> **Test** -> **Copy to registry**. The full post-merge suite takes **several hours per framework** due to model downloads, GPU inference, and multi-GPU tests.

| Stage | What it does | Local equivalent |
|-------|-------------|-----------------|
| Build image | Render Dockerfile, build runtime container | `container/render.py --framework=vllm --target=runtime && docker build ...` |
| Sanity check | Verify packages are installed in the image | `docker run --rm <image> /workspace/deploy/sanity_check.py --runtime-check --no-gpu-check` |
| CPU-only tests (parallel) | `(pre_merge or post_merge) and <framework> and gpu_0` | `pytest -m "(pre_merge or post_merge) and vllm and gpu_0" -n auto --dist=loadscope -v --tb=short` |
| Single GPU tests (sequential) | `(pre_merge or post_merge) and <framework> and gpu_1` | `pytest -m "(pre_merge or post_merge) and vllm and gpu_1" -v --tb=short` |
| Multi-GPU tests (sequential) | `(pre_merge or post_merge) and <framework> and (gpu_2 or gpu_4)` | `pytest -m "(pre_merge or post_merge) and vllm and (gpu_2 or gpu_4)" -v --tb=short` |

### Nightly (daily at midnight PST)

Same structure as post-merge but selects tests marked `nightly` instead of `(pre_merge or post_merge)`:
```bash
pytest -m "nightly and vllm and gpu_1" -v --tb=short
```

### Reproducing CI locally

All commands shown in the "Local equivalent" columns above are also documented in [Running Rust Checks and Tests](#running-rust-checks-and-tests) and [Running Python Tests](#running-python-tests). Run Rust commands from the repo root, repeating for each workspace dir: `.`, `lib/bindings/python`, `lib/runtime/examples`, `lib/bindings/kvbm`. Run Python commands inside a container.

---

## Additional Requirements

### Flaky Tests

Tests must be deterministic. A flaky test -- one that sometimes passes and sometimes fails without code changes -- wastes CI time and erodes developer trust in the test suite. If you encounter or introduce a flaky test:

1. **Fix it first.** Remove sources of non-determinism: set a fixed random seed, eliminate race conditions, mock network calls, avoid relying on execution order.
2. **If a fix is not immediately possible**, quarantine the test to prevent it from blocking other developers:
   - `@pytest.mark.skip(reason="Flaky: <ticket link>")` -- disables the test entirely. Use when the test provides no signal in its current state.
   - `@pytest.mark.xfail(reason="Flaky: <ticket link>", strict=False)` -- runs the test but does not fail the suite. Use when you still want visibility into pass/fail rates while you investigate.
   - In Rust, use `#[ignore]` with a comment explaining why.
3. **File a ticket** for every quarantined test. Flaky tests without an owner drift indefinitely.
4. **Do not leave tests quarantined for more than one sprint.** If the root cause is elusive, delete the test and rewrite it.

### Timeouts

Long-running tests **must** have an explicit timeout. A test that hangs (e.g., waiting for a model server that never starts, or a deadlocked subprocess) will block the entire CI job and waste GPU-hours for everyone.

- Use the `pytest-timeout` plugin (already in our dependencies):
  ```python
  @pytest.mark.timeout(300)  # 5 minutes
  def test_e2e_inference():
      ...
  ```
- Set the timeout to **2x-3x the observed average runtime**. This gives enough headroom for legitimate variance (model loading jitter, CPU contention) while still catching genuine hangs. For example, if a test normally completes in 90 seconds, set `@pytest.mark.timeout(240)`.
- For Rust, use `#[timeout(Duration::from_secs(300))]` or set a default timeout in `Cargo.toml`.
- In CI, the workflow also enforces a global job timeout (see workflow YAML files). Per-test timeouts catch problems earlier and with a clearer error message than a blanket job cancellation.

### Time Budgets

- If a test exceeds its time budget (see [Test Types and Locations](#test-types-and-locations)), profile it with `pytest --durations=0` and consider mocking heavy dependencies, using a smaller model checkpoint, or moving it to a nightly/weekly pipeline with `@pytest.mark.slow`.

### Time Budget Industry Practices

Our per-test time targets are informed by widely adopted test size classifications:

- **Bazel test sizes** assign concrete timeouts by size: small = 60 s, medium = 300 s (5 min), large = 900 s (15 min), enormous = 3600 s (1 hr). Tests exceeding their size's expected range trigger warnings. ([Bazel Test Encyclopedia](https://docs.bazel.build/versions/2.0.0/test-encyclopedia.html))
- **Software Engineering at Google** (Winters, Manshreck, Wright, 2020) classifies tests by resource scope: small tests run in a single process with no I/O; medium tests run on a single machine; large tests may span machines. Google targets roughly 80% unit / 15% integration / 5% E2E by test count. ([Ch. 11](https://abseil.io/resources/swe-book/html/ch11.html))
- **Practitioner benchmarks** (Fowler, Seemann) suggest unit tests at 1-10 ms each, integration tests at ~100 ms, and E2E tests at ~1 s for non-GPU workloads. A TDD-cycle unit suite should complete in under 10 seconds. ([Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html), [TDD in 10 seconds](https://blog.ploeh.dk/2012/05/24/TDDtestsuitesshouldrunin10secondsorless/))

GPU and model-loading overhead means Dynamo E2E tests are inherently slower than typical web-service E2E tests. Model load time alone is often 30-120 s for large models, which is why our E2E budget is 5 minutes rather than 1 second.

---

## Troubleshooting

- If a test is not running, verify the filename, markers, and folder location.
- For flaky tests, see [Flaky Tests](#flaky-tests) above. Fix, quarantine with `skip`/`xfail`, and file a ticket.
- For slow or hanging tests, add `@pytest.mark.timeout()` (see [Timeouts](#timeouts)) and profile with `pytest --durations=0`.
- If model downloads fail, ensure `HF_TOKEN` is set and network access is available.
- If `ImportError: cannot import name ... from 'dynamo._internal'`, you need to compile the Rust bindings first (see [Prerequisites](#prerequisites)).
- If coverage is insufficient, add more tests or refactor code for better testability.

---

## GPU VRAM Profiler (`profile_pytest.py`)

When writing or reviewing GPU tests, use `tests/utils/profile_pytest.py` to measure how much VRAM a test actually needs. The script runs the test repeatedly with different GPU memory caps and uses binary search to find the minimum VRAM required. It then prints recommended pytest markers you can copy into your test.

### How it works

The profiler automatically detects the engine type and uses the appropriate binary search:

- **vLLM**: bisects `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` (bytes) → `--kv-cache-memory-bytes`. Finds the minimum KV cache bytes where the test passes, applies a 2x safety factor. Outputs `profiled_vram_gib` and `requested_vllm_kv_cache_bytes` markers.
- **SGLang**: bisects `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` (token count) → `--max-total-tokens`. Finds the minimum KV cache tokens where the test passes, applies a 2x safety factor, then runs a final probe at the safe token count to measure the actual VRAM. Outputs `profiled_vram_gib` and `requested_sglang_kv_tokens` markers.
- **TRT-LLM**: bisects `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` (token count) → `KvCacheConfig.max_tokens` via `--override-engine-args` JSON. Same logic as SGLang (token-based bisection, 2x safety). Outputs `profiled_vram_gib` and `requested_trtllm_kv_tokens` markers. For non-text models (video/image diffusion) that don't use KV cache, use `--no-find-min-vram` for a single-pass VRAM measurement — binary search won't work because the model doesn't log KV token allocation.

**Requirement (vLLM):** The launch script must honor `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES`. This is handled by `build_vllm_gpu_mem_args` in `gpu_utils.sh` (returns `--kv-cache-memory-bytes N`).

**Requirement (SGLang):** The launch script must honor `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS`. This is handled by `build_sglang_gpu_mem_args` in `gpu_utils.sh` (returns `--max-total-tokens N`).

**Requirement (TRT-LLM):** The launch script must honor `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` (and optionally `_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES`). This is handled by `build_trtllm_override_args_with_mem` in `gpu_utils.sh` (returns JSON for `--override-engine-args`). Note: this is a separate function from `build_vllm_gpu_mem_args` / `build_sglang_gpu_mem_args` because TRT-LLM requires JSON merging.

**Requirement (all engines):** Do not hardcode `CUDA_VISIBLE_DEVICES` in launch scripts. The profiler and parallel test runner set `CUDA_VISIBLE_DEVICES` to pin each test to a specific GPU. A script that overrides this (e.g. `CUDA_VISIBLE_DEVICES=0`) will ignore the assignment and land on the wrong GPU. Instead, inherit from the environment with a default:

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

Then pass the variable to each worker: `CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m dynamo.vllm ...`. For multi-GPU scripts that assign distinct GPUs per worker, use named env vars with defaults (e.g. `PREFILL_CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-0}"`).

### Engine-specific mapping

Launch scripts call engine-specific functions from `examples/common/gpu_utils.sh` which check env var overrides and return the appropriate CLI flags:

```bash
# vLLM
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &

# SGLang
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &

# TRT-LLM (requires JSON merging, separate function)
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
python -m dynamo.trtllm --model-path "$MODEL" ${OVERRIDE_JSON:+--override-engine-args "$OVERRIDE_JSON"} &
```

Env vars control engine allocation during profiling and parallel test execution:

**`_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES`** (integer) — vLLM only:

| Engine  | Returned CLI flag                | Notes |
|---------|----------------------------------|-------|
| vLLM    | `--kv-cache-memory-bytes N`      | Exact byte cap on KV cache; deterministic and parallel-safe |

**`_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS`** (integer) — SGLang only:

| Engine  | Returned CLI flag                | Notes |
|---------|----------------------------------|-------|
| SGLang  | `--max-total-tokens N`           | Token-based KV cache cap |

**`_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS`** (integer) — TRT-LLM text models:

| Engine  | Returned JSON                                          | Notes |
|---------|--------------------------------------------------------|-------|
| TRT-LLM | `{"kv_cache_config": {"max_tokens": N}}`              | Token-based KV cache cap via `--override-engine-args` |

**`_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES`** (integer) — TRT-LLM non-text models:

| Engine  | Returned JSON                                                    | Notes |
|---------|------------------------------------------------------------------|-------|
| TRT-LLM | `{"kv_cache_config": {"max_gpu_total_bytes": N}}`               | Byte-based cap via `--override-engine-args`. For diffusion models. |

All use absolute caps — deterministic and independent of current free memory, which is critical for parallel test execution. See `examples/common/gpu_utils.md`.

### Usage

```bash
# vLLM: binary search for minimum KV cache bytes
python tests/utils/profile_pytest.py tests/serve/test_vllm.py::test_serve_deployment[aggregated] -xvs

# Profile on a specific GPU (default: 0)
python tests/utils/profile_pytest.py --gpu 1 tests/serve/test_vllm.py::test_serve_deployment[aggregated] -xvs

# SGLang: binary search for minimum KV cache tokens (automatic)
python tests/utils/profile_pytest.py tests/serve/test_sglang.py::test_sglang_deployment[aggregated-2] -xvs

# TRT-LLM: binary search for minimum KV cache tokens (text models)
python tests/utils/profile_pytest.py tests/serve/test_trtllm.py::test_deployment[aggregated-2] -xvs

# TRT-LLM: single-pass for diffusion models (no KV cache, binary search won't work)
python tests/utils/profile_pytest.py --no-find-min-vram tests/serve/test_trtllm.py::test_deployment[video_diffusion-2] -xvs

# Single-pass profiling (no binary search, just measure one run using default RAM)
python tests/utils/profile_pytest.py --no-find-min-vram tests/serve/test_vllm.py::test_serve_deployment[aggregated]
```

### Example output (vLLM)

```bash
========================================================================
FIND MINIMUM KV CACHE BYTES (vLLM, deterministic) (binary search)
========================================================================
  GPU total : 48.0 GiB
  GPU free  : 47.4 GiB  (in use: 0.6 GiB)
  Test      : tests/serve/test_vllm.py::test_serve_deployment[aggregated] -x

  [probe 1] Validation run: kv_cache=23296 MiB (50% of free)
  [PASS] peak 2.9 GiB, wall 42s, iter took 49s
  ...
  [probe 6/15] kv_cache=449 MiB (471,027,000 bytes)
  [PASS] peak 2.9 GiB, wall 41s, iter took 49s

  [probe 7/15] kv_cache=224 MiB (235,513,856 bytes)
  [FAIL] OOM, iter took 30s

========================================================================
  Minimum KV cache : 449 MiB (471,027,000 bytes)
  Safe KV cache    : 898 MiB (942,054,000 bytes) (2x safety)
  Peak VRAM        : 2.9 GiB

  Recommended markers:
    @pytest.mark.profiled_vram_gib(2.9)
    @pytest.mark.requested_vllm_kv_cache_bytes(942_054_000),  # KV cache cap (2x safety over min=471_027_000)
========================================================================

========================================================================
Recommended markers to add to your pytest. You can copy-paste this:
========================================================================
# Measured using: tests/utils/profile_pytest.py tests/serve/test_vllm.py::test_serve_deployment[aggregated]
@pytest.mark.e2e  # wall time 41.2s, loads a real model
@pytest.mark.gpu_1  # 1 GPU(s) used, peak 2.9 GiB
@pytest.mark.profiled_vram_gib(2.9)  # actual nvidia-smi peak
@pytest.mark.requested_vllm_kv_cache_bytes(942_054_000)  # KV cache cap (2x safety over min=471_027_000)
@pytest.mark.timeout(124)  # 3x observed 41.2s

  WARNING: Wall time 41.2s is too slow for pre_merge (> 20s). Consider post_merge or nightly instead.
========================================================================
```

### Example output (SGLang — token-based bisection)

```bash
========================================================================
FIND MINIMUM KV TOKENS (SGLang) (binary search)
========================================================================
  GPU total : 48.0 GiB
  GPU free  : 47.4 GiB  (in use: 0.6 GiB)
  Test      : tests/serve/test_sglang.py::test_sglang_deployment[aggregated-2] -xvs

  [probe 1] Validation run (no token cap)
  [PASS] peak 43.0 GiB, wall 36s, max_total_tokens=366688, iter took 44s
  ...
  [probe 14/15] tokens=48  [~1 left, ETA ~45s]
  [PASS] tokens=48, peak 3.7 GiB, wall 26s, iter took 34s
  [final probe] Measuring VRAM at safe_tokens=96
  [PASS] tokens=96, peak 3.7 GiB, wall 27s

========================================================================
MINIMUM KV TOKENS RESULT
========================================================================
  Minimum tokens  : 16 (raw bisection result)
  Recommended     : 96 (2x safety)
  Peak VRAM       : 3.7 GiB (at 96 tokens)
  @pytest.mark.profiled_vram_gib(3.7)
  @pytest.mark.requested_sglang_kv_tokens(96),  # KV cache cap (2x safety over min=48)
========================================================================
```

### Example output (TRT-LLM — token-based bisection)

```bash
========================================================================
FIND MINIMUM KV TOKENS (TensorRT-LLM) (binary search)
========================================================================
  GPU total : 48.0 GiB
  GPU free  : 47.1 GiB  (in use: 0.9 GiB)
  Test      : tests/serve/test_trtllm.py::test_deployment[aggregated-2] -xvs

  [probe 1] Validation run (no token cap, default fraction)
  [PASS] peak 41.3 GiB, wall 48s, max_tokens=41472 (TensorRT-LLM), iter took 56s
  ...
  [probe 6/12] tokens=1296
  [PASS] tokens=1296, peak 3.7 GiB, wall 46s, iter took 54s
  [EARLY STOP] Peak VRAM stable for last 3 probes
  [final probe] Measuring VRAM at safe_tokens=2592
  [PASS] tokens=2592, peak 3.9 GiB, wall 46s

========================================================================
MINIMUM KV TOKENS RESULT (TensorRT-LLM)
========================================================================
  Minimum tokens  : 1296 (raw bisection result)
  Recommended     : 2592 (2x safety)
  Peak VRAM       : 3.9 GiB (at 2592 tokens)
  @pytest.mark.profiled_vram_gib(3.9)
  @pytest.mark.requested_trtllm_kv_tokens(2592),  # KV cache cap (2x safety over min=1296)
========================================================================
```

### How to use the recommendations

1. **Copy the `@pytest.mark.*` lines** into your test function or `pytestmark` list.

2. **VRAM markers** — `profiled_vram_gib(N)` records the actual nvidia-smi peak (for filtering/scheduling), `requested_vllm_kv_cache_bytes(N)` or `requested_sglang_kv_tokens(N)` controls the engine's KV cache allocation for deterministic parallel execution. Use `--max-vram-gib=N` to deselect tests whose profiled VRAM exceeds N (see [Filtering by VRAM](#filtering-by-vram)). The WARNING lines in the profiler output tell you which GPU tiers would be too small (e.g., "Will OOM on T4 (16 GiB)").

3. **Lifecycle markers** — the profiler recommends `pre_merge` only for tests under 20 seconds. For slower tests, it warns you to consider `post_merge` or `nightly` but does not choose for you — use your judgment based on how critical the test is for catching regressions early.

4. **Timeout** — the recommended value is 3x the observed wall time. Adjust upward if your test has high variance (e.g., first-run model download, flaky network).

5. **Test type** (`unit`, `integration`, `e2e`) — inferred from wall time and whether a real model was loaded. Override if you know better (e.g., a fast test that uses a mock engine is `integration`, not `e2e`).

### Options

| Flag | Description |
|------|-------------|
| `--kv-bytes` | No-op (kept for backward compat). vLLM always bisects on `--kv-cache-memory-bytes` |
| `--no-find-min-vram` | Skip binary search; run a single profiling pass instead |
| `--interval N` | GPU sampling interval in seconds (default: 1.0) |
| `--baseline-seconds N` | Seconds to sample before launching pytest (default: 3.0) |
| `--teardown-seconds N` | Seconds to sample after pytest exits (default: 5.0) |
| `--csv FILE` | Write raw nvidia-smi samples to a CSV file |
| `--no-recommend` | Suppress marker recommendations |

---

## References
- [pytest documentation](https://docs.pytest.org/en/stable/)
- [Bazel Test Encyclopedia — test sizes and timeouts](https://docs.bazel.build/versions/2.0.0/test-encyclopedia.html)
- [Software Engineering at Google — Testing Overview (Ch. 11)](https://abseil.io/resources/swe-book/html/ch11.html)
- [Martin Fowler — The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Mark Seemann — TDD test suites should run in 10 seconds or less](https://blog.ploeh.dk/2012/05/24/TDDtestsuitesshouldrunin10secondsorless/)

For further assistance, contact the Dynamo development team.
