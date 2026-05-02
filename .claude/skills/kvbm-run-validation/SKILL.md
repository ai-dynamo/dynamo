---
name: kvbm-run-validation
description: Run KVBM accuracy/determinism validation tests (container or local, auto-detects or asks)
user-invocable: true
disable-model-invocation: true
---

# Run KVBM Validation Tests

Run KVBM integration tests to validate accuracy, determinism, and correctness. Supports two execution modes:

- **local**: Runs pytest against the `.sandbox/` venv. Fast iteration loop. Requires `/dynamo:kvbm:sandbox-venv` + `/dynamo:kvbm:maturin-dev` first.
- **container**: Runs pytest inside a dynamo vllm image (hermetic, mirrors CI). Requires a built image — see `/dynamo:kvbm:build`.

This skill **detects or asks** which mode to use. It does not hardcode a default.

For the faster three-shell local iteration flow (deps / server / eval — skip this wrapper), see `/dynamo:kvbm:decomposed-run`.

## Arguments

`/dynamo:kvbm:run-validation [scope] [--spec SPEC_ID] [--fast] [--mode local|container] [--image IMAGE] [--enable-mla]`

- **scope** (default: `quick`):
  - `quick` — Pre-merge marker (`-m "kvbm and pre_merge"`). 1 GPU. ~5 min.
  - `agg-v1` — v1 determinism only (spec ids starting with `v1-`). ~15 min.
  - `agg-v2-intra` — v2 intra-onboard determinism only. ~15 min.
  - `agg-v2-inter` — v2 inter-onboard determinism only. ~15 min.
  - `agg` — all v1 + v2 (intra + inter) determinism specs. ~45 min.
  - `disagg` — `test_determinism_disagg.py`. 2 GPUs. ~15 min.
  - `full` — All KVBM tests (`-m "kvbm or kvbm_concurrency"`). ~30+ min.
  - `<filename>` — Run a specific file (e.g. `test_chunked_prefill.py`).
- **--spec SPEC_ID** — Run a single parametrization by id (e.g. `v1-Qwen3-0.6B`). Maps to `-k $SPEC_ID`. Overrides scope marker filters.
- **--fast** — `KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2`.
- **--mode local|container** — Force a mode. When omitted, this skill probes and asks.
- **--image IMAGE** (default: `dynamo:latest-vllm`) — Container image (container mode only).
- **--enable-mla** — Set `KVBM_ENABLE_MLA=1` to unlock `DeepSeek-V2-Lite` specs.

## Step 0: Detect Or Ask Mode

**Always run the probe unless `--mode` was passed explicitly.**

```bash
# Local mode viable?
LOCAL_OK=no
if [ -x .sandbox/bin/python ] && .sandbox/bin/python -c "import kvbm" 2>/dev/null; then
    LOCAL_OK=yes
fi

# Container mode viable?
CONTAINER_OK=no
if command -v docker >/dev/null 2>&1; then
    if docker image inspect "${IMAGE:-dynamo:latest-vllm}" >/dev/null 2>&1; then
        CONTAINER_OK=yes
    fi
fi
```

Decision table:

| LOCAL_OK | CONTAINER_OK | Action |
|---|---|---|
| yes | yes | **Ask** the user which mode to use. Show both. |
| yes | no | Use local. Announce: "local mode selected (no usable container image)". |
| no | yes | Use container. Announce: "container mode selected (no `.sandbox/` venv)". |
| no | no | **Stop**. Tell the user: `run /dynamo:kvbm:sandbox-venv + /dynamo:kvbm:maturin-dev for local, or /dynamo:kvbm:build for container, then retry`. |

If `--mode local` or `--mode container` is passed, skip the probe but still verify the chosen mode's prerequisite and bail with a clear pointer if missing.

## Step 1: Map Scope To Pytest Args

Spec ids are read from `_CACHE_RESET_SPECS` in `tests/kvbm_integration/test_determinism_agg.py`. Authoritative list via:

```bash
.sandbox/bin/python - <<'PY'
from tests.kvbm_integration.test_determinism_agg import _CACHE_RESET_SPECS
for s in _CACHE_RESET_SPECS: print(s.id)
PY
```

| Scope | Pytest args | GPUs | Est. time |
|---|---|---|---|
| `quick` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm and pre_merge"` | 1 | ~5 min |
| `agg-v1` | `tests/kvbm_integration/test_determinism_agg.py -k "v1-"` | 1 | ~15 min (8B) / ~3 min (0.6B) |
| `agg-v2-intra` | `tests/kvbm_integration/test_determinism_agg.py -k "v2-" -k "intra"` | 1 | ~15 min / ~3 min |
| `agg-v2-inter` | `tests/kvbm_integration/test_determinism_agg.py -k "v2-" -k "inter"` | 1 | ~15 min / ~3 min |
| `agg` | `tests/kvbm_integration/test_determinism_agg.py` | 1 | ~45 min / ~10 min |
| `disagg` | `tests/kvbm_integration/test_determinism_disagg.py` | 2 | ~15 min |
| `full` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm or kvbm_concurrency"` | 1-2 | ~30+ min |
| `<file>` | `tests/kvbm_integration/<file>` | varies | varies |

If `--spec SPEC_ID` was provided, use `-k $SPEC_ID` instead of the scope filter.

**Gotcha**: spec ids containing `Qwen3-0.6B` only exist when `KVBM_MODEL_ID=Qwen/Qwen3-0.6B` is set. The default first model is `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.

## Step 2: Show Plan And Confirm

```
KVBM Validation Plan
────────────────────
Mode:      <local|container>
Scope:     <scope>
Spec:      <spec-id or "scope filter">
Model:     <KVBM_MODEL_ID or "per-spec default">
MLA gate:  <enabled|disabled>
Fast mode: <yes|no>
GPUs:      <count>
Est. time: <estimate>
[if container] Image: <image>

Command:
  <full command preview>
```

For `agg` without `--fast`, suggest `--fast` and offer the decomposed flow:

> Tip: `--fast` drops this to <5 min. For the three-shell iteration loop (iterate eval without re-spawning vllm), use `/dynamo:kvbm:decomposed-run <spec-id> --fast` instead.

Confirm before proceeding.

## Step 3: Build Environment Variables

Always:
```
RUST_BACKTRACE=1
```

Conditional:
```
# --fast
KVBM_MAX_ITERATIONS=2
KVBM_NUM_ITERATIONS=2
KVBM_REQUEST_DELAY=2

# --enable-mla
KVBM_ENABLE_MLA=1

# KVBM_MODEL_ID — if the caller wants a Qwen spec id, this MUST be set
KVBM_MODEL_ID=Qwen/Qwen3-0.6B
```

On GB10, the phase-3 reference knobs for Qwen3-0.6B:
```
KVBM_CPU_BLOCKS=2000
KVBM_GPU_BLOCKS=512
KVBM_GPU_MEMORY_UTILIZATION=0.5
KVBM_SERVER_START_TIMEOUT=600
```

## Step 4: Run Tests

Timeouts:
- `quick`: 600s
- `agg-*` fast: 600s
- `agg-*` full: 1800s (more for 8B specs — leave the per-test timeout at the computed value in `test_determinism_agg.py`)
- `disagg`: 1800s
- `full`: 3600s
- single file/spec: 900s

### Local mode

```bash
RUST_BACKTRACE=1 <env_vars> \
    timeout <timeout> .sandbox/bin/python -m pytest <pytest_args> -v --tb=short -s
```

Tests reuse host NATS/etcd on defaults (4222 / 2379) if reachable (aligned with `conftest.py:runtime_services` in phase 3). Otherwise fixtures spawn them.

### Container mode

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

Note: do NOT use `--network host` (conflicts with host NATS/etcd), `--runtime nvidia` (not portable), or `-it` (hangs in non-interactive shells). The runtime image is missing `pytest-benchmark` / `pytest-asyncio`, so install them at container startup.

## Step 5: Report Results

```
KVBM Validation Results
───────────────────────
Mode:      <local|container>
Passed:    X
Failed:    Y
Skipped:   Z
Errors:    W
Duration:  Nm Ns
```

If failures, show `--tb=short` output and point at `/dynamo:kvbm:diagnose`:

```
For per-test log analysis and live /metrics inspection:
  /dynamo:kvbm:diagnose
```

Stack health quick-check patterns (grep the newest per-test log under `/tmp/dynamo_tests/`):
- `KvConnectorWorker initialized` — worker bootstrapped
- `Auto-detected device layout` — tensor layout OK
- `ConnectorLeader initialized with onboard mode onboard_mode=Intra|Inter` — v2 leader + mode
- `Application startup complete` — vllm ready
- `kvbm_offload_blocks_d2h > 0` in `/metrics` — offload active

## Reference: Test Files

| File | Tests | Markers | GPUs |
|---|---|---|---|
| `test_kvbm.py` | offload_and_onboard, gpu_cache_eviction, onboarding_determinism | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_chunked_prefill.py` | chunked prefill offload | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_kvbm_vllm_integration.py` | vLLM interface assumptions | kvbm, integration, gpu_0, vllm, nightly, pre_merge | 0 |
| `test_consolidator_router_e2e.py` | consolidator + router E2E | kvbm, e2e, slow, gpu_1, pre_merge | 1 |
| `test_determinism_agg.py` | cache_reset, concurrent load | e2e, slow, gpu_1, nightly | 1 |
| `test_determinism_disagg.py` | disagg determinism | kvbm, vllm, trtllm, e2e, slow, gpu_2, nightly | 2 |
| `test_cuda_graph.py` | CUDA graph (TRT-LLM only) | kvbm, trtllm, nightly, gpu_1 | 1 |

## Reference: Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KVBM_MODEL_ID` | DeepSeek-R1-Distill-Llama-8B (1st spec only) | Override the first `_MODEL_CONFIGS` entry; required for Qwen spec ids |
| `KVBM_CPU_BLOCKS` | 10000 | CPU cache block count |
| `KVBM_GPU_BLOCKS` | 2048 | GPU cache block count |
| `KVBM_MAX_ITERATIONS` | 100 | Max iterations (cache-reset test) |
| `KVBM_NUM_ITERATIONS` | 15 | Number of iterations (concurrent test) |
| `KVBM_REQUEST_DELAY` | 30 | Delay between iterations (seconds) |
| `KVBM_ENABLE_MLA` | unset | Unlock DeepSeek-V2-Lite specs |
| `KVBM_SERVER_START_TIMEOUT` | 600 | Server startup timeout |
| `KVBM_GPU_MEMORY_UTILIZATION` | per-spec | vllm memory fraction |
| `KVBM_EXTERNAL_BASE_URL` | unset | External-attach mode (set by `run_server.sh`) |
| `KVBM_EXTERNAL_METRICS_PORT` | unset | External-attach mode |
| `KVBM_SPEC_ID` | unset | Spec id handshake for decomposed flow |

## Related Skills

- `/dynamo:kvbm:sandbox-venv` — set up `.sandbox/` (local mode prerequisite)
- `/dynamo:kvbm:maturin-dev` — rebuild kvbm-py3 (local mode prerequisite)
- `/dynamo:kvbm:decomposed-run` — three-shell iteration loop (bypasses this wrapper)
- `/dynamo:kvbm:diagnose` — read-only triage of failed runs
- `/dynamo:kvbm:build` — build the container image (container mode prerequisite)
- `/dynamo:kvbm:rebuild-lockfiles` — regenerate all Cargo.lock files
