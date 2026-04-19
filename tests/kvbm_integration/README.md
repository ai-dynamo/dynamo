# KVBM Determinism & Behavior Tests

## Overview

This suite validates that vLLM + KVBM produces deterministic outputs across
prefix-cache resets and under concurrent load. The suite is decomposed into
three composable layers so local iteration can run them in separate shells
and re-run the eval loop without re-spawning vLLM.

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **A. Deps** | `fixtures/deps.py` | Bring up NATS + etcd for v1 (reuse-or-spawn). v2 agg is a no-op (discovery defaults to `None` per `lib/kvbm-config/src/messenger.rs:43`). |
| **B. Server** | `fixtures/server.py` | Launch `vllm serve` with the kv-transfer-config built by `build_kv_transfer_config(version, model_config)`. |
| **C. Eval** | `fixtures/eval.py` | Bind `AggDeterminismTester` to the running server and run the determinism loop. |

`test_determinism_agg.py` parametrizes a single `kvbm_server_spec` axis that
bundles `(kvbm_version, model_config, cpu_blocks, gpu_blocks, onboard_mode)`.
As of phase 5 the list enumerates `("v1", "v2")` crossed with the model list;
v2 is further crossed with both onboard modes (`intra` and `inter`), so every
v2 spec appears twice — once per mode — to catch mode-specific regressions on
every run. Spec ids take the form `v1-<model>` and `v2-<model>-<mode>`.

## Running — composed (one shell)

```bash
# Quick smoke (2 iterations instead of 100)
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
    pytest tests/kvbm_integration/test_determinism_agg.py \
        -v -k "test_determinism_agg_with_cache_reset" --tb=short

# Full run (~6 minutes)
pytest tests/kvbm_integration/test_determinism_agg.py -v -s
```

## Running — decomposed (three shells, fast iteration)

The decomposed flow is keyed on a **spec id** that matches
`KvbmServerSpec.id` in `test_determinism_agg.py` — e.g.
`v1-DeepSeek-R1-Distill-Llama-8B`. `run_server.sh` reconstructs the
exact spec from the test module's parametrize list, so attention
backend, block size, and `batch_invariant` always come from the
canonical source. Scripts never hardcode per-model attributes.

```bash
# ─── v1 flow (needs NATS + etcd) ───────────────────────────────────────

# Shell 1 — bring up NATS + etcd (reuses if already reachable, else spawns
# in the foreground and prints exports; Ctrl-C to stop the spawned form)
bash tests/kvbm_integration/scripts/run_deps_v1.sh

# Shell 2 — export the env vars printed by shell 1, then launch vllm
# for one specific spec id from the test parametrize list.
export NATS_SERVER=nats://localhost:NNNN
export ETCD_ENDPOINTS=http://localhost:NNNN
bash tests/kvbm_integration/scripts/run_server.sh v1-DeepSeek-R1-Distill-Llama-8B

# Shell 3 — export the env vars printed by shell 2 (note the third one,
# KVBM_SPEC_ID — it filters the eval to the spec shell 2 launched).
export KVBM_EXTERNAL_BASE_URL=http://localhost:NNNN
export KVBM_EXTERNAL_METRICS_PORT=NNNN
export KVBM_SPEC_ID=v1-DeepSeek-R1-Distill-Llama-8B
bash tests/kvbm_integration/scripts/run_eval.sh

# ─── v2 flow (no NATS + etcd needed — agg discovery defaults to None) ──

# Shell 1 for v2 is a no-op: the v2 leader's `discovery` field defaults
# to None (lib/kvbm-config/src/messenger.rs:43) and `build_messenger`
# short-circuits. Explicitly unset any v1 deps vars that may be in your
# env so you don't accidentally inherit a previous shell's exports:
unset NATS_SERVER ETCD_ENDPOINTS

# Shell 2 — launch vllm for a v2 spec id. Each v2 spec carries its
# onboard mode in the id (`intra` or `inter`).
KVBM_MODEL_ID=Qwen/Qwen3-0.6B \
    bash tests/kvbm_integration/scripts/run_server.sh v2-Qwen3-0.6B-intra

# Shell 3 — same pattern as v1.
export KVBM_EXTERNAL_BASE_URL=http://localhost:NNNN
export KVBM_EXTERNAL_METRICS_PORT=NNNN
export KVBM_SPEC_ID=v2-Qwen3-0.6B-intra
bash tests/kvbm_integration/scripts/run_eval.sh
```

`run_eval.sh` defaults to running `test_determinism_agg_with_cache_reset`
filtered by `KVBM_SPEC_ID`. Positional args override entirely (ad-hoc
runs).

`run_server.sh` honors `KVBM_CPU_BLOCKS` / `KVBM_GPU_BLOCKS` /
`KVBM_SERVER_START_TIMEOUT` env overrides (applied via
`dataclasses.replace` on the canonical spec). For v2 specs, `KVBM_CPU_BLOCKS`
drives `cache.host.num_blocks` directly (exact parity with v1's
`DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` path). The Rust leader bails
at startup if neither a host nor a disk cache tier is configured.

`v1-DeepSeek-V2-Lite` and `v2-DeepSeek-V2-Lite-{intra,inter}` are the suite's
MLA specs and are currently gated; set `KVBM_ENABLE_MLA=1` (in both the
server shell and the eval shell) to opt in. Pytest reports them as
`SKIPPED` otherwise.

## External-attach mode

Setting `KVBM_EXTERNAL_BASE_URL` makes the `kvbm_server` and `kvbm_deps`
fixtures skip spawn and bind to a long-lived external server. Both v1 deps
bring-up *and* vllm spawn are bypassed; the test loop runs against the
existing process. This is the contract the three-shell flow above relies on.

`KVBM_EXTERNAL_METRICS_PORT` is required alongside (the metrics endpoint is
read by the determinism eval). Both env vars must be set together.

## Markers

- `kvbm` — KV behavior and model determinism tests
- `kvbm_concurrency` — concurrent-load variant
- `e2e` — end-to-end tests
- `slow` — long-running (~minutes)
- `gpu_1` — needs one GPU
- `nightly` — preferred for nightly runs

## Configuration

Environment variables control server settings and test load.

### Server / model
- `KVBM_MODEL_ID` (default: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- `KVBM_SERVER_PORT` — pin the vllm port; otherwise dynamically allocated
- `KVBM_SERVER_START_TIMEOUT` (default: `600`s) — large MLA models need 600s+
- `KVBM_GPU_MEMORY_UTILIZATION` (default: `0.9`)
- `KVBM_MLA_BACKEND` (default: `TRITON_MLA`) — set `FLASH_ATTN_MLA` on H100

### Cache size overrides
- `KVBM_CPU_BLOCKS` (default: `10000` for cache-reset, `30000` for concurrent)
- `KVBM_GPU_BLOCKS` (default: `2048`)

### Test duration
- `KVBM_MAX_ITERATIONS` (default: `100`) — cache-reset test
- `KVBM_NUM_ITERATIONS` (default: `15`) — concurrent test
- `KVBM_REQUEST_DELAY` (default: `30`s)
- `KVBM_HTTP_TIMEOUT` (default: `30`s)

### External-attach mode
- `KVBM_EXTERNAL_BASE_URL` — when set, fixture skips spawn; required in shell 3
- `KVBM_EXTERNAL_METRICS_PORT` — required alongside `KVBM_EXTERNAL_BASE_URL`
- `KVBM_SPEC_ID` — required by `run_eval.sh` (the spec id `run_server.sh`
  was launched with). Filters the pytest run to that one parametrization.

### MLA gate
- `KVBM_ENABLE_MLA` — opt in to running MLA specs (currently only
  `v1-DeepSeek-V2-Lite`). Default unset; MLA specs are pytest-skipped
  and `run_server.sh` refuses to launch them. Set to `1`/`true`/`yes`/`on`
  to enable.

### Quick-iteration example

```bash
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
    pytest tests/kvbm_integration/test_determinism_agg.py -v --tb=short
```

## Requirements

- `vllm` executable on `PATH`
- `kvbm.vllm_integration.connector` resolves to v1 (phase-1 back-compat shim)
- NATS + etcd binaries available *or* a reachable NATS/etcd already running
  (the v1 deps layer reuses pre-existing services when possible)
- One GPU for the cache-reset test; vLLM bench installed for the concurrent test

## Notes

- Logs are written under the per-test directory from `tests/conftest.py`
  (`resolve_test_output_path`) and include the vLLM stdout/stderr.
- Warmup is critical — disabling it hides initialization-related determinism bugs.
- As of phase 5 the parametrize list enumerates both v1 and v2. Every v2 spec
  is crossed with both onboard modes (`intra`, `inter`), so a single run exercises
  both paths and catches mode-specific regressions immediately.

## Other tests in this directory

- `test_kvbm.py`, `test_chunked_prefill.py` — use `llm_server_kvbm` from
  `common.py` (a `ManagedProcess`-based fixture). They are **not** affected
  by the three-layer decomposition above. Consolidating them with `kvbm_server`
  is a future cleanup.
- `test_consolidator_router_e2e.py`, `test_kvbm_vllm_integration.py`,
  `test_cuda_graph.py` — use their own server bring-up patterns.
- `test_determinism_disagg.py` — defines its own (disagg) `LLMServerManager`;
  not migrated in phase 2 (agg-only scope).
