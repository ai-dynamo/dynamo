---
name: kvbm-run-validation
description: Run KVBM accuracy/determinism validation tests (container or local)
user-invocable: true
disable-model-invocation: true
---

# Run KVBM Validation Tests

Run KVBM integration tests to validate accuracy, determinism, and correctness.

## Arguments

`/dynamo:kvbm:run-validation [scope] [--model MODEL_ID] [--fast] [--image IMAGE] [--local]`

- **scope** (default: `quick`):
  - `quick` — Pre-merge tests (`-m "kvbm and pre_merge"`). 1 GPU. ~5 min.
  - `agg` — Aggregated determinism (`test_determinism_agg.py`). 1 GPU. ~15 min full / ~3 min fast.
  - `disagg` — Disaggregated determinism (`test_determinism_disagg.py`). 2 GPUs. ~15 min.
  - `full` — All KVBM tests (`-m "kvbm or kvbm_concurrency"`). 1-2 GPUs. ~30+ min.
  - `<filename>` — Run a specific file (e.g., `test_chunked_prefill.py`).
- **--model MODEL_ID** — Override default model via `KVBM_MODEL_ID` env var.
- **--fast** — Use reduced iteration counts for `agg`/`disagg` scope.
- **--image IMAGE** (default: `dynamo:latest-vllm`) — Container image to use. Ignored with `--local`.
- **--local** — Run tests directly on host using `.sandbox` venv instead of a container. Requires NATS and etcd running on host.

## Step 0: Determine Execution Mode

If `--local` is specified (or no `--image` is provided and `.sandbox` venv exists), use **local mode**. Otherwise use **container mode**.

**Local mode prerequisites** — check these before proceeding:

```bash
# 1. Check .sandbox venv exists with kvbm installed
.sandbox/bin/python -c "import kvbm; print(f'kvbm {kvbm.__version__}')"

# 2. Check nats-server binary is available (tests start their own instances)
which nats-server

# 3. Check etcd binary is available (tests start their own instances)
which etcd
```

**NATS/etcd port behavior:** Tests start their own NATS/etcd instances via pytest fixtures
using dynamically allocated ports (4223+/2380+). This is safe alongside host NATS/etcd
running on default ports (4222/2379).

If `.sandbox` doesn't exist or kvbm isn't installed, set it up:

```bash
uv venv .sandbox --clear
VIRTUAL_ENV=.sandbox PATH=.sandbox/bin:$PATH uv pip install \
    -r tests/kvbm_integration/requirements.txt \
    lib/bindings/kvbm/target/wheels/kvbm-*.whl
```

If no kvbm wheel exists yet, build one first:

```bash
cd lib/bindings/kvbm && maturin build --release --out target/wheels && cd -
```

**vllm install:** Most KVBM tests require vllm. Read the pinned version from `pyproject.toml`
(under `[project.optional-dependencies] vllm`) and install that exact version:

```bash
# Check pyproject.toml for the pinned version, e.g. vllm==0.19.0
grep 'vllm\[' pyproject.toml
# Install the matching version
VIRTUAL_ENV=.sandbox PATH=.sandbox/bin:$PATH uv pip install "vllm[flashinfer]==<pinned_version>"
```

Always verify the installed version matches the Dynamo pin — a version mismatch will cause
subtle test failures. Tests that require vllm will skip gracefully if it's not installed.

## Step 1: Parse Arguments and Determine Test Scope

Map the scope to the pytest command:

| Scope | Pytest args | GPUs | Est. time |
|-------|-------------|------|-----------|
| `quick` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm and pre_merge"` | 1 | ~5 min |
| `agg` | `tests/kvbm_integration/test_determinism_agg.py` | 1 | 15 min / 3 min fast |
| `disagg` | `tests/kvbm_integration/test_determinism_disagg.py` | 2 | ~15 min |
| `full` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm or kvbm_concurrency"` | 1-2 | ~30+ min |
| `<file>` | `tests/kvbm_integration/<file>` | varies | varies |

### What each scope runs

**quick** (pre_merge marker):
- `test_kvbm.py`: test_offload_and_onboard, test_gpu_cache_eviction, test_onboarding_determinism
- `test_chunked_prefill.py`: chunked prefill offload validation
- `test_kvbm_vllm_integration.py`: vLLM interface assumption tests (gpu_0)
- `test_consolidator_router_e2e.py`: KV event consolidator + router E2E

**agg** (test_determinism_agg.py):
- `test_determinism_agg_with_cache_reset`: Determinism across offload→reset→onboard (DeepSeek-R1-8B, DeepSeek-V2-Lite)
- `test_concurrent_determinism_under_load`: IFEval prompts under concurrent load (needs `kvbm_concurrency` marker)

**disagg** (test_determinism_disagg.py):
- Same as agg but with disaggregated prefill-decode on 2 GPUs. Threshold: >=95% match.

## Step 2: Show Plan and Confirm

Present to the user:

**Local mode:**
```
KVBM Validation Plan
────────────────────
Mode:      local (.sandbox venv)
Scope:     <scope>
GPUs:      <count>
Est. time: <estimate>
Model:     <override or "default per test">
Fast mode: <yes/no>

Command:
  RUST_BACKTRACE=1 .sandbox/bin/python -m pytest <pytest_args> -v --tb=short -s
```

**Container mode:**
```
KVBM Validation Plan
────────────────────
Mode:      container
Scope:     <scope>
Image:     <image>
GPUs:      <count>
Est. time: <estimate>
Model:     <override or "default per test">
Fast mode: <yes/no>

Command:
  docker run --gpus all --rm --shm-size=10G --ulimit memlock=-1 \
    --ulimit stack=67108864 --ulimit nofile=65536:65536 \
    -e RUST_BACKTRACE=1 -e HF_TOKEN \
    -v $(pwd):/workspace --cap-add CAP_SYS_PTRACE --ipc host \
    -w /workspace <image> \
    timeout <timeout> pytest <pytest_args> -v --tb=short -s
```

Note: Do NOT use `--network host` (conflicts with host NATS/etcd) or `--runtime nvidia`
(not universally available; `--gpus all` is sufficient). Do NOT use `-it` flag when running
from a non-interactive shell.

For `agg` scope without `--fast`, suggest it:
> Tip: Add `--fast` to reduce agg test time from ~15 min to ~3 min (uses KVBM_MAX_ITERATIONS=2).

Confirm before proceeding.

## Step 3: Build Environment Variables

Always set `RUST_BACKTRACE=1`.

If `--model` provided: set `KVBM_MODEL_ID=<model>`.

If `--fast` (or user accepts fast mode suggestion):
```
KVBM_MAX_ITERATIONS=2
KVBM_NUM_ITERATIONS=2
KVBM_REQUEST_DELAY=2
```

## Step 4: Run Tests

Timeout values:
- `quick`: 600 (10 min)
- `agg` fast: 600 (10 min)
- `agg` full: 1800 (30 min)
- `disagg`: 1800 (30 min)
- `full`: 3600 (60 min)
- specific file: 900 (15 min)

### Local mode

Run pytest directly using the `.sandbox` venv. Prepend env vars inline.

```bash
RUST_BACKTRACE=1 <extra_env_vars> \
    timeout <timeout> .sandbox/bin/python -m pytest <pytest_args> -v --tb=short -s
```

Tests start their own NATS/etcd via fixtures using dynamic ports (safe alongside host
NATS/etcd on default ports).

### Container mode

Run via docker with `--gpus all`. Do NOT use `--runtime nvidia`, `--network host`, or `-it`.

```bash
docker run --gpus all --rm \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    -e RUST_BACKTRACE=1 \
    -e HF_TOKEN \
    <extra_env_flags> \
    -v $(pwd):/workspace \
    -v /tmp:/tmp \
    -v /mnt/:/mnt \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    -w /workspace \
    <image> \
    bash -c "pip install pytest-benchmark pytest-asyncio -q && timeout <timeout> pytest <pytest_args> -v --tb=short -s"
```

Note: The runtime image is missing some test deps (`pytest-benchmark`, `pytest-asyncio`),
so install them at container startup. Tests manage their own NATS/etcd servers via the
`runtime_services` fixture when running in the container.

## Step 5: Report Results

Parse the pytest output and present:

```
KVBM Validation Results
───────────────────────
Passed:  X
Failed:  Y
Skipped: Z
Errors:  W
Duration: Nm Ns
```

If failures occurred, show the `--tb=short` output for each and suggest next steps:
- Determinism failures: "Check KVBM offload/onboard metrics. Re-run with `-e DYN_LOG=debug` for detail."
- Server startup failures: "Check GPU memory with `nvidia-smi`. Try increasing `KVBM_SERVER_START_TIMEOUT`."
- Import errors: "Verify the container image has kvbm and vllm installed: `docker run --rm <image> python -c 'import kvbm; import vllm'`"

## Reference: Test Files

| File | Tests | Module markers | GPUs |
|------|-------|----------------|------|
| `test_kvbm.py` | offload_and_onboard, gpu_cache_eviction, onboarding_determinism | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_chunked_prefill.py` | chunked prefill offload | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_kvbm_vllm_integration.py` | vLLM interface assumptions | kvbm, integration, gpu_0, vllm, nightly, pre_merge | 0 |
| `test_consolidator_router_e2e.py` | consolidator + router E2E | kvbm, e2e, slow, gpu_1, pre_merge | 1 |
| `test_determinism_agg.py` | cache_reset determinism, concurrent load | e2e, slow, gpu_1, nightly (methods add kvbm/kvbm_concurrency) | 1 |
| `test_determinism_disagg.py` | disagg determinism with cache reset | kvbm, vllm, trtllm, e2e, slow, gpu_2, nightly | 2 |
| `test_cuda_graph.py` | CUDA graph (TRT-LLM only) | kvbm, trtllm, nightly, gpu_1 | 1 |

## Reference: Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KVBM_MODEL_ID` | per-test | Override model |
| `KVBM_CPU_BLOCKS` | 10000 | CPU cache block count |
| `KVBM_GPU_BLOCKS` | 2048 | GPU cache block count |
| `KVBM_MAX_ITERATIONS` | 100 | Max iterations (determinism tests) |
| `KVBM_NUM_ITERATIONS` | 15 | Number of iterations (concurrent test) |
| `KVBM_REQUEST_DELAY` | 30 | Delay between iterations (seconds) |
| `KVBM_MAX_TOKENS` | per-test | Max tokens to generate |
| `KVBM_SEED` | per-test | Random seed |
| `KVBM_SERVER_START_TIMEOUT` | 600 | Server startup timeout (seconds) |
