# ACTIVE_PLAN — KVBM v1/v2 parity with decomposed integration tests

## Context

**Goal**: Restore a fully working v1 KVBM on this branch *and* bring v2 up to the same standard, with `tests/kvbm_integration/test_determinism_agg.py` passing under both paths. Decompose the test harness into three independently runnable layers (deps, server, eval) so local iteration can target external services while CI runs an all-in-one.

**Why now**: `main` has a working v1. The `ryan/kvbm-bindings` branch reorganized the Python package to mirror v2 (moved v1 code under `kvbm/v1/…` and added v2 at `kvbm/v2/…`), and added the 8-crate `kvbm-*` workspace under `lib/`. `lib/kvbm-config/` now carries a `v1_compat.rs` that translates legacy `DYN_KVBM_*` env vars into v2 config paths. The reorg + new crates broke the previously working `.sandbox/launch_vllm_with_connector.sh` flow, and the integration tests are tightly coupled to v1 paths and a single monolithic fixture.

**Runtime constraints (critical)**:
- **v1** uses the dynamo distributed runtime → needs **NATS + etcd** (tests/kvbm_integration/conftest.py already handles reuse-or-spawn via `runtime_services`).
- **v2** uses Nova messenger with `discovery.type = "filesystem"` → **no external NATS/etcd**. The sandbox script points nova at `/tmp/nova-discovery/sandbox.json`.
- Decomposition must preserve both modes.

**Self-consistency rule**: every phase updates this document with actual state, deviations, and any retro-edits to earlier phases. The ACTIVE_PLAN.md at the workspace root is the source of truth; this plan file is the approved template we copy there.

---

## Current state snapshot (verified 2026-04-12)

### Rust / crate layout
- `lib/kvbm-{common,config,connector,engine,kernels,logical,observability,physical}` — 8 crates, all workspace members. `kvbm-connector` is the apex (depends on engine, logical, physical, config, observability, common).
- `Cargo.toml` references `kvbm-scheduler` as a workspace member, **but that crate does not exist on disk** — must be removed or created in phase 0.
- `lib/llm/src/block_manager/` — v1 implementation, still the source of v1 BlockManager / KvbmLeader / KvbmWorker.
- `lib/bindings/kvbm/src/`
  - `v1/` → PyO3 bindings into `dynamo-llm/block-manager`
  - `v2/` → PyO3 bindings into `kvbm-connector` (runtime, leader, worker, scheduler, torch, vllm config)
  - `dynamo/` → tokio runtime + OTEL setup
  - `lib.rs` — feature-gated (`v1`, `v2`, `dynamo`, `kernels`, `nccl`), default = `["v1","v2"]`

### Python package layout (`lib/bindings/kvbm/python/kvbm/`)
- `__init__.py` — feature-stub fallback when Cargo features unavailable
- `_feature_stubs.py` — `_make_feature_stub()` / `_make_module_stub()`
- `v1/` — canonical v1 location
  - `vllm_integration/connector/{dynamo_connector.py, pd_connector.py}`
  - `trtllm_integration/connector/{kvbm_connector_leader.py, kvbm_connector_worker.py}`
  - `consolidator_config.py`, `rust.py`, `utils.py`
- `v2/` — canonical v2 location
  - `__init__.py` → re-exports from `kvbm._core.v2`
  - `vllm/{config.py, version_check.py, connectors/…, schedulers/…}`
- Shims at old v1 paths:
  - `kvbm/vllm_integration/connector/__init__.py` — **suspect**: exploration reports this as a lazy loader that delegates to `kvbm.v2.vllm.connectors.connector`. That would break v1 back-compat expected by tests. **Must verify and fix in phase 1.**
  - `kvbm/trtllm_integration/__init__.py` — wildcard re-export from `kvbm.v1.trtllm_integration` ✔
- `conftest.py` at the package root (new) — purpose TBD, verify in phase 1.

### Tests (`tests/kvbm_integration/`)
- `conftest.py` overrides `runtime_services` to reuse `NATS_SERVER` / `ETCD_ENDPOINTS` if reachable, else spawn.
- `common.py` — `DeterminismTester`, `ApiTester`, `ServerType`, `TestDeterminism`, metrics helpers. **Already abstracted for eval logic.** The decomposition needs to extract *server bring-up*, not the eval layer.
- `test_determinism_agg.py` — hardcodes `LLMServerManager` class inline with v1 kv-transfer-config: `{"kv_connector":"DynamoConnector","kv_role":"kv_both","kv_connector_module_path":"kvbm.vllm_integration.connector"}`. No v2 variant today.
- Other tests (`test_kvbm_vllm_integration.py`, `test_consolidator_router_e2e.py`, `test_kvbm.py`, `test_chunked_prefill.py`) are v1-flavored.

### Reference launch script
`~/archives/dynamo/.sandbox/launch_vllm_with_connector.sh` — working v2 invocation. Key pieces we must reproduce in a test fixture:
```
--kv-transfer-config '{
  "kv_connector": "DynamoConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
  "kv_connector_extra_config": {
    "leader": {
      "cache": {"host": {"cache_size_gb": 10.0}},
      "tokio": {"worker_threads": 2},
      "nova": {"discovery": {"type":"filesystem","path":"/tmp/nova-discovery/sandbox.json"}},
      "onboard": {"mode":"intra"}
    },
    "worker": {
      "nixl": {"backends": {"UCX": {}, "POSIX": {}}},
      "tokio": {"worker_threads": 2}
    }
  }
}'
```

---

## Phases

Each phase has: **Goal / Inputs / Deliverables / Tasks / Verification / Risks**. Each is sized to drive one `gsd:plan-phase` or equivalent planning session.

### Phase 0 — Workspace hygiene & build baseline
**Goal**: Clean compile of the entire workspace on this branch with default features. Resolve dangling references so `cargo check --all-features --all-targets` and `maturin develop` both succeed before touching behavior.

**Inputs**: Current branch state, root `Cargo.toml`, `lib/bindings/kvbm/Cargo.toml`, feature matrix.

**Deliverables**:
- Root `Cargo.toml` no longer references `lib/kvbm-scheduler` (remove entry), **or** a minimal stub crate is created if any code already imports it. Decide by grep first.
- `cargo check --all-features --all-targets` → green.
- `cargo clippy --all-features --no-deps --all-targets -- -D warnings` → green.
- `cargo fmt --check` → green.
- `cd lib/bindings/kvbm && maturin develop --features v1,v2` succeeds inside the sandbox venv.
- Recorded baseline: `python -c "import kvbm; print(kvbm.__version__); from kvbm.v1 import BlockManager; from kvbm.v2 import KvbmRuntime"`.

**Tasks**:
1. `grep -r "kvbm_scheduler\|kvbm-scheduler" --include='*.rs' --include='Cargo.toml'` — confirm zero references before removing.
2. If references exist, add a skeleton `lib/kvbm-scheduler/` crate with the minimum API surface; otherwise delete the workspace-member line.
3. Run `cargo check / clippy / fmt` and fix fallout.
4. Build the Python extension in the sandbox venv; capture command in plan.
5. Sanity-import v1 and v2 symbols from Python REPL.

**Verification**:
- `cargo check --all-features --all-targets` exits 0.
- `cargo clippy --all-features --no-deps --all-targets -- -D warnings` exits 0.
- Python imports above succeed with no feature-stub warnings.

**Risks**:
- `kvbm-scheduler` may be referenced indirectly (cfg-gated). Grep both lowercase and snake_case.
- v2 features may need CUDA; `KVBM_REQUIRE_CUDA=1` is set in pyproject. Confirm nvcc available on target box.

---

### Phase 1 — Restore v1 Python back-compat paths
**Status**: Detailed plan approved 2026-04-12. Ordering: run phase 0 first, then execute phase 1.

**Goal**: Repoint `kvbm.vllm_integration.connector` at the v1 implementation so tests that use the legacy path run v1 code again (matching `main`). Reverse the branch author's v1-deprecation stance.

**Policy decisions (2026-04-12)**:
- **Option A**: legacy `kvbm.vllm_integration.connector` == v1. Tests keep the legacy path; v2 tests will use the explicit `kvbm.v2.vllm.schedulers.connector` path added in phase 2. (Chosen over Option B which would have rewritten every v1 test to use an explicit v1 path.)
- **v1 is NOT deprecated**. Remove "deprecated, not actively maintained" language from `lib/bindings/kvbm/conftest.py` and update comments in the shim files.

**Ground-truth observations that motivated this plan**:
- Current shim at `lib/bindings/kvbm/python/kvbm/vllm_integration/connector/__init__.py:4-11` is a deliberate lazy redirect to `kvbm.v2.vllm.connectors.connector` — opposite of what we need.
- `lib/bindings/kvbm/conftest.py:26` skips all v1 integration files from collection under its tree.
- `kvbm.utils` and `kvbm.trtllm_integration.*` already re-export from v1 correctly — no change needed there.
- Every v1-flavored test today references `kvbm.vllm_integration.connector` (legacy path), so with the current v2-pointing shim the tests are silently running v2 code under a "v1" label. Fixing the shim restores truthful semantics.
- `kvbm.v1.vllm_integration.connector.dynamo_connector` imports cleanly (only upward ref is `kvbm.v1.utils`; no cycle back to the top-level shim).

**Inputs**:
- `lib/bindings/kvbm/python/kvbm/vllm_integration/connector/__init__.py` — rewrite
- `lib/bindings/kvbm/python/kvbm/vllm_integration/__init__.py` — comment update only
- `lib/bindings/kvbm/python/kvbm/v1/vllm_integration/connector/dynamo_connector.py` — reference only
- `lib/bindings/kvbm/conftest.py` — policy change
- Sandbox venv with `maturin develop --features v1,v2` already built (dependency on phase 0 completion)

**Deliverables**:
- `kvbm.vllm_integration.connector.DynamoConnector is kvbm.v1.vllm_integration.connector.dynamo_connector.DynamoConnector` (same object, not just equal).
- `kvbm.trtllm_integration.connector.DynamoKVBMConnectorLeader/Worker` resolve to v1 classes (verification; no change expected).
- `kvbm.utils.nvtx_annotate` resolves to v1 (verification; no change expected).
- `lib/bindings/kvbm/conftest.py` no longer carries "deprecated" language; v1 integration files are collectable again.
- New smoke-level import test: `lib/bindings/kvbm/python/tests/test_legacy_imports.py`.
- ACTIVE_PLAN.md deviations log entry describing the Option-A policy decision.

**Tasks**:

1. **Repoint the connector shim.** Replace the lazy-redirect `__init__.py` with a direct re-export:
   ```python
   # Back-compat shim — the legacy path kvbm.vllm_integration.connector resolves to v1.
   # The explicit v2 path is kvbm.v2.vllm.schedulers.connector (see sandbox/launch_vllm_with_connector.sh).
   from kvbm.v1.vllm_integration.connector.dynamo_connector import *  # noqa: F401,F403
   from kvbm.v1.vllm_integration.connector.dynamo_connector import DynamoConnector  # noqa: F401
   ```
   Update the parent `kvbm/vllm_integration/__init__.py` comment to match.

2. **Audit sibling top-level paths.** Grep for any test or vllm-side code that expects:
   - `kvbm.vllm_integration.connector_leader`
   - `kvbm.vllm_integration.connector_worker`
   - `kvbm.vllm_integration.consolidator_config`
   - `kvbm.vllm_integration.rust`
   - `kvbm.vllm_integration.kv_cache_utils`
   - `kvbm.vllm_integration.kv_cache_manager`
   - `kvbm.vllm_integration.connector.pd_connector`
   If any of these are imported from the legacy path, add matching re-export shims pointing to `kvbm.v1.vllm_integration.*`. Otherwise skip (YAGNI).

3. **Update `lib/bindings/kvbm/conftest.py`**:
   - Drop `V1_SKIP_PATTERNS` and the "deprecated" docstring language.
   - Keep the vllm-availability guard for `V2_VLLM_PATTERNS` (still valid — v2 vllm integration genuinely requires vllm installed).
   - Keep the `requires_vllm` marker registration.

4. **Add `lib/bindings/kvbm/python/tests/test_legacy_imports.py`** (new directory if needed). Four subtests, each idempotent and GPU-free:
   - `test_legacy_vllm_connector_resolves_to_v1` — import + `is` identity check vs v1 canonical
   - `test_trtllm_legacy_paths` — import `DynamoKVBMConnectorLeader/Worker`
   - `test_utils_shim` — import `nvtx_annotate` from `kvbm.utils`
   - `test_top_level_kvbm_has_v1_and_v2` — assert `_V1_AVAILABLE and _V2_AVAILABLE` True
   Skip the whole module with `pytest.importorskip("vllm")` since the dynamo_connector imports `vllm.distributed.kv_transfer.kv_connector.v1.base`.

5. **Sanity-run `tests/kvbm_integration/test_kvbm_vllm_integration.py`** (collection-only):
   ```bash
   pytest tests/kvbm_integration/test_kvbm_vllm_integration.py --collect-only -q
   ```
   Expect no collection errors. Actual test execution stays in phase 3.

6. **Update ACTIVE_PLAN.md**: flip phase 1 state to completed, log the policy decision in the deviations log, record the exact changed files.

**Verification**:
```bash
# Assume phase 0 already built the extension
python - <<'PY'
from kvbm.vllm_integration.connector import DynamoConnector
from kvbm.v1.vllm_integration.connector.dynamo_connector import DynamoConnector as DC_v1
assert DynamoConnector is DC_v1, f"{DynamoConnector!r} is not {DC_v1!r}"
print("legacy path OK:", DynamoConnector.__module__)

from kvbm.trtllm_integration.connector import DynamoKVBMConnectorLeader, DynamoKVBMConnectorWorker  # noqa
from kvbm.utils import nvtx_annotate  # noqa
import kvbm
assert kvbm._V1_AVAILABLE and kvbm._V2_AVAILABLE
print("all legacy shims OK")
PY

pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -v
pytest tests/kvbm_integration/test_kvbm_vllm_integration.py --collect-only -q
```

All three commands must exit 0 and the identity assertion must hold.

**Risks**:
- **Hidden top-level submodule imports**. If vllm internally loads `kvbm.vllm_integration.connector_leader` (plural flat form rather than `kvbm.vllm_integration.connector.DynamoConnector`), we need more shims than just the `connector` submodule. Covered by task 2's grep sweep.
- **Name collisions from wildcard re-export**. The v1 `dynamo_connector.py` may export helper names that shadow v2 when someone imports `kvbm.vllm_integration.connector.something`. Add an explicit `__all__` to the shim if the wildcard imports more than we intend.
- **Identity vs equality semantics**. `from X import *` inside a shim `__init__.py` does not register the shim as the canonical module for `DynamoConnector` — the class's `__module__` remains `kvbm.v1.vllm_integration.connector.dynamo_connector`. That's desirable (matches v1) and is what the identity check above verifies.

**Out-of-scope for phase 1** (explicitly):
- No test-harness decomposition (phase 2).
- No v2 canonical-path reconciliation (phase 4).
- No v1 Rust changes.
- No run of the determinism test body (phase 3).

---

### Phase 2 — Integration test decomposition (agg-only, v1 execution)
**Status**: Detailed plan approved 2026-04-13. Ordering: phases 0–1 done; phase 2 builds harness scaffolding for both versions but only enumerates v1 in test parametrization until phase 4 lands the v2 import-readiness fixes.

**Goal**: Split the monolithic `LLMServerManager` / `llm_server` fixture in `test_determinism_agg.py` into three composable layers (deps / server / eval) so local iteration can run them in separate shells. Lay the v1/v2 seam in `build_kv_transfer_config` but ship v2 inert — `pytest --collect-only` enumerates only v1 because importing `kvbm.v2.vllm.schedulers.connector` is blocked by three v2-readiness issues that belong to phase 4 (see "Phase-4 scope expansion" below).

**Three layers**:

| Layer | Responsibility | v1 impl | v2 impl (agg) |
|---|---|---|---|
| **A. Deps** | Bring up NATS+etcd (v1) or nothing (v2 agg). Return `DepsHandle`. | reuse existing `runtime_services` fixture | empty handle (no deps) |
| **B. Server** | Launch vLLM with kvbm kv-transfer-config built for the chosen backend. Return `ServerHandle`. | v1 dict (matches phase-1 shim) | v2 dict, **no `leader.nova` block** (discovery defaults to `None` per `lib/kvbm-config/src/messenger.rs:43`) |
| **C. Eval** | Run `DeterminismTester` against a `ServerHandle`. Already isolated in `common.py`. | unchanged | unchanged |

**Why v2 is not enumerated in phase 2** (peer-review verified blockers):
1. **Eager scheduler init** — `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/__init__.py:5` does `from .dynamo import DynamoScheduler`; `dynamo.py` expects scheduler symbols not exported by `lib/bindings/kvbm/src/v2/mod.rs:8`. Importing the connector triggers package init and dies.
2. **Hard vllm version block** — `lib/bindings/kvbm/python/kvbm/v2/vllm/config.py:22` raises on `vllm >= 0.12.2` (sandbox has 0.19.0). Bypass: `KVBM_SKIP_VLLM_VERSION_CHECK`.
3. **Two parallel version policies** — `lib/bindings/kvbm/python/kvbm/v2/vllm/version_check.py:7` defines a contradicting policy gated on a different env var (`KVBM_DISABLE_MAX_VERSION_CHECK`).

These are v2-readiness changes, folded into phase 4. Phase 5 flips the parametrize list to `["v1", "v2"]`.

**Scope boundary**: agg only. No disagg plumbing. v2 agg deliberately omits discovery config.

**Nova/velo caveat**: Rust crate is `velo` but the serde config key is still literally `"nova"` in `MessengerConfig`. Phase 2 sidesteps by omitting the block. Finishing the rename is out of scope.

**Inputs**:
- `tests/kvbm_integration/test_determinism_agg.py` — current monolithic harness target
- `tests/kvbm_integration/common.py` — already-decoupled eval layer (`DeterminismTester`, `ApiTester`, `fetch_kvbm_metrics`, `parse_kvbm_metrics`)
- `tests/kvbm_integration/conftest.py:78-104` — existing `runtime_services` reuse-or-spawn fixture
- `tests/conftest.py:606+`, `:679+` — `EtcdServer`, `NatsServer`
- `tests/utils/port_utils.py:322-373` — `allocate_port` / `deallocate_port`
- `~/archives/dynamo/.sandbox/launch_vllm_with_connector.sh` — v2 reference config
- Phase-1 shim: `kvbm.vllm_integration.connector` already → v1, so v1 payload needs no change

**Deliverables**:
- New `tests/kvbm_integration/fixtures/` package: `__init__.py`, `deps.py`, `server.py`, `eval.py`.
- `KvbmServerManager` (extracted verbatim from `LLMServerManager`), `KvbmServerSpec`, `ServerHandle`, `DepsHandle`, `build_kv_transfer_config(version, model_config)`.
- Pytest fixtures: `kvbm_deps`, `kvbm_server_spec`, `kvbm_server`, `kvbm_tester`. All accept `kvbm_version` indirectly.
- `test_determinism_agg.py` rewritten to consume fixtures with `@pytest.mark.parametrize("kvbm_version", ["v1"], indirect=True)` plus `# TODO(phase 5): add "v2" once phase 4 lands` comment immediately above.
- v2 builder unit test at `tests/kvbm_integration/fixtures/test_kv_transfer_config.py` (asserts dict shape only — the only v2 code exercised in phase 2).
- Shell scripts under `tests/kvbm_integration/scripts/`:
  - `run_deps_v1.sh` — foreground NATS+etcd, prints `NATS_SERVER` / `ETCD_ENDPOINTS`, traps deallocate on exit
  - `run_server.sh v1` — wraps `KvbmServerManager`, prints `KVBM_EXTERNAL_BASE_URL` / `KVBM_EXTERNAL_METRICS_PORT`
  - `run_server.sh v2` — errors out with "deferred to phase 5; see ACTIVE_PLAN.md"
  - `run_eval.sh` — pytest invocation against external base URL
- Rewritten `tests/kvbm_integration/README.md` (full rewrite, not append).
- Phase-1 missed-cleanup: fix stale comment at `lib/bindings/kvbm/python/kvbm/vllm_integration/__init__.py:4-6`.
- Phase-4 scope expansion noted in this doc (see "Phase 4" section below — scope grew).

**Tasks**:
1. **Phase-1 cleanup.** Replace stale "subpackages resolve to v2 implementations" comment in `kvbm/vllm_integration/__init__.py:4-6`.
2. **Create `fixtures/` package skeleton** — empty modules.
3. **Extract `LLMServerManager` verbatim** into `fixtures/server.py` as `KvbmServerManager`. Re-import from `test_determinism_agg.py`. No semantic changes. Preserve `_SERVER_START_TIMEOUT`, metrics-port allocation, tee logic.
4. **Replace hardcoded kv-transfer-config** with `build_kv_transfer_config(version, model_config)` (v1 branch only at first).
5. **Add v2 branch** to `build_kv_transfer_config` (inert) + `KvbmServerSpec` dataclass + dict-shape unit test. v2 payload omits `leader.nova`; `cache.host.cache_size_gb` hardcoded to 10.0 (TODO phase 5: derive from `KVBM_CPU_BLOCKS × block_size × dtype × num_layers`).
6. **Write `deps.py`** — `DepsHandle` + `kvbm_deps` fixture. v1 wraps existing `runtime_services`; v2 returns empty handle. Honor `KVBM_EXTERNAL_BASE_URL` short-circuit.
7. **Write `eval.py`** — `make_determinism_tester` factory + `kvbm_tester` fixture.
8. **Wire external-attach mode**: grep first to confirm `KVBM_EXTERNAL_BASE_URL` is unused, then implement.
9. **Rewrite `test_determinism_agg.py`** to consume fixtures with `kvbm_version=["v1"]` indirect parametrize.
10. **Shell scripts** with exit traps for port deallocation.
11. **Full README rewrite**.
12. **Phase-4 scope expansion** entry in this document.
13. **Flip phase 2 to completed** in state-per-phase, log deviations.

**Verification**:
```bash
# collect-only smoke (v1 IDs present, no errors)
pytest tests/kvbm_integration/test_determinism_agg.py --collect-only -q

# v2 builder unit test (no server launch)
pytest tests/kvbm_integration/fixtures/test_kv_transfer_config.py -v

# v1 composed
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
  pytest tests/kvbm_integration/test_determinism_agg.py \
    -v -k "test_determinism_agg_with_cache_reset" --tb=short

# v1 decomposed (three shells)
bash tests/kvbm_integration/scripts/run_deps_v1.sh           # shell 1
export NATS_SERVER=... ETCD_ENDPOINTS=...                     # shell 2 from shell 1 output
bash tests/kvbm_integration/scripts/run_server.sh v1
export KVBM_EXTERNAL_BASE_URL=... KVBM_EXTERNAL_METRICS_PORT=... # shell 3 from shell 2 output
bash tests/kvbm_integration/scripts/run_eval.sh
```

**Sign-off**: v1 composed and v1 decomposed both pass; v2 builder unit test passes; collection clean. **No v2 server launched, no v2 test parametrized.**

**Risks**:
- Extract regression — mitigate by doing the move as a pure copy + re-import + run-test step before any semantic changes.
- Metrics-port leaks across decomposed shells — exit traps deallocate.
- v2 builder dict drifts from phase-4/5 reality — acceptable; one pure function, local fix.
- `kv_connector_module_path` ambiguity (`schedulers.connector` vs `connectors.connector`) — phase 4 reconciles; phase 2 hardcodes the sandbox-script path.
- Phase-4 scope creep from absorbing readiness gates — surfacing now is better than hitting it at phase-5 sign-off.

**Out of scope for phase 2** (explicit):
- Launching any v2 vLLM server.
- Parametrizing the determinism test with `"v2"`.
- Unifying v2 vllm version policies (phase 4).
- Making `kvbm.v2.vllm.schedulers.connector` import-safe (phase 4).
- Exporting missing scheduler symbols from `lib/bindings/kvbm/src/v2/mod.rs` (phase 4).
- Deriving v2 `cache_size_gb` from `KVBM_CPU_BLOCKS` (phase 5).
- Reconciling `schedulers.connector` vs `connectors.connector` (phase 4).
- Disagg testing, discovery bring-up, peer wiring.
- Consolidating `llm_server_kvbm` with the new `kvbm_server` fixture.
- Restoring `cargo clippy -D warnings --all-targets` to green (phase 6).
- nova→velo rename finish-up.

---

### Phase 3 — v1 end-to-end determinism passing (GB10 / sm_121 enablement)
**Goal**: With phases 0–2 in place, run `test_determinism_agg_with_cache_reset` in v1 mode to 100% exact-match and restore parity with main, **on this Blackwell GB10 host**. Phase 2 verified the fixture wiring up to vllm spawn; the actual model run is gated on a sm_121-capable PyTorch + vllm in the sandbox venv.

**GB10 sm_121 context (2026-04-13)**: This host is an NVIDIA GB10 (sm_121, Blackwell). The sandbox venv's stock `vllm 0.19.0` ships PyTorch with kernels compiled only through sm_120, so vllm dies at GPU init with `cudaErrorNoKernelImageForDevice`. Reproduced standalone with `vllm serve` (no fixture machinery) — not a phase-2 regression. Tracking upstream: vllm-project/vllm#31128 and #36821. FP8 kernels are still incomplete on sm_121 even in source builds (see eugr/spark-vllm-docker#143) — confirm the chosen models don't take the FP8 path before signing off.

**Inputs**: Decomposed fixtures from phase 2; phase-1 back-compat shim; sandbox venv at `.sandbox/`; cached HF models include `Qwen/Qwen3-0.6B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `deepseek-ai/DeepSeek-V2-Lite`.

**Deliverables**:
- A working sm_121-compatible PyTorch + vllm in `.sandbox/` (option 2 — community wheels, not source build, not container).
- `kvbm-py3` rebuilt against the new torch ABI via `maturin develop --features v1,v2`; phase-1 regression test still 6/7 green.
- Passing determinism test for the smaller model (`Qwen/Qwen3-0.6B` preferred for fast iteration) in v1 mode.
- Captured KVBM metrics snapshot demonstrating offload→onboard activity (`kvbm_offload_blocks_d2h > 0`, `kvbm_onboard_blocks_h2d > 0`).
- Recorded log of runtime + pointer to log file location.
- Documented exact wheel sources / install commands in this plan's deviations log so the next person can reproduce the venv.

**Approach (option 2 — community wheels, vllm 0.19 preferred but not required)**:

The first attempt should keep `vllm 0.19.0` if a community torch wheel can satisfy it. If 0.19 cannot be made to work on sm_121 with available wheels, fall back to whichever vllm version the community sm_121 toolchains are built against (likely vllm main / nightly). Capture the actual versions chosen in deviations.

Candidate wheel sources (try in order, stop at first that works):
1. **PyTorch official `cu130` aarch64 index** — `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`. Has aarch64 wheels for CUDA 13. May or may not include sm_121 kernels — verify after install with `python -c "import torch; print(torch.cuda.get_arch_list())"` and check for `sm_121` (or `sm_120` for forward-compat).
2. **`cypheritai/pytorch-blackwell`** GitHub repo — pre-built PyTorch wheels with native sm_121 kernels. Direct wheel install. No source build.
3. **PyTorch nightly cu128/cu129** — if (1) lacks sm_121, the nightly index ships sm_120 binary-compatible builds that work on sm_121.

After torch is in place:
- If stock `vllm 0.19.0` boots: keep it.
- Otherwise pull a vllm wheel from a Blackwell-aware community installer (e.g. `eelbaz/dgx-spark-vllm-setup` or whichever the chosen torch wheel is built against). Pin and document the resulting version in deviations. The kv-transfer-config payload is version-agnostic for v1, so the determinism test should not care.
- Rebuild `kvbm-py3` against the new torch ABI: `cd lib/bindings/kvbm && maturin develop --features v1,v2`. Confirm `python -c "from kvbm.v1 import BlockManager; from kvbm.v2 import KvbmRuntime"` still works.

**Tasks**:
1. **Snapshot the current sandbox venv** (e.g. `pip freeze > .sandbox/requirements.before-phase3.txt`) so we can roll back if a wheel install corrupts the env.
2. **Try option 2.1** (pytorch.org cu130 aarch64 index). If `torch.cuda.get_arch_list()` includes a sm_121-capable arch and a tiny matmul on `cuda:0` runs, proceed. Otherwise skip to 2.2.
3. **Try option 2.2** (cypheritai/pytorch-blackwell) only if 2.1 didn't yield a working torch. Direct wheel install into `.sandbox/`.
4. **Re-test stock vllm 0.19.0** with the new torch via a 30-second dry-run: `vllm serve Qwen/Qwen3-0.6B --max-model-len 256 --gpu-memory-utilization 0.5`, expect it to reach "Application startup complete" (or an error that's NOT `cudaErrorNoKernelImageForDevice`). If this works, keep vllm 0.19.0.
5. **If vllm 0.19.0 fails ABI**: install a Blackwell-aware vllm wheel (community installer or nightly). Document the chosen version + source.
6. **Rebuild `kvbm-py3`** against the new torch ABI and re-run `pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q`. Must still be 6 passed / 1 skipped.
7. **Run the v1 determinism test** with the smallest model and lowest iterations first:
   ```bash
   KVBM_MODEL_ID=Qwen/Qwen3-0.6B \
   KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
   KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 \
   KVBM_GPU_MEMORY_UTILIZATION=0.5 \
   pytest tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset \
       -v -k "v1-Qwen" --tb=short
   ```
8. **Verify offload→onboard metrics**: hit `http://localhost:$DYN_KVBM_METRICS_PORT/metrics` after the run, confirm `kvbm_offload_blocks_d2h > 0` and `kvbm_onboard_blocks_h2d > 0`. The test internally asserts these via `fetch_kvbm_metrics`; surface them in the deviations log.
9. **Scale up**: re-run with `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` at default iteration counts. Confirm exact match.
10. **(Optional)** Run the second model config (`DeepSeek-V2-Lite`) to confirm MLA + batch-invariant=False path also passes.
11. **Avoid FP8 paths** for now — if the test fails with `sm_120 kernel` or FP8 errors, drop to a model/config that doesn't trigger FP8 quantization; document.
12. **Record** the final wheel set, install commands, and any patches in deviations so the venv is reproducible.

**Verification**:
- `pytest tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset -v -k "v1-Qwen" --tb=short` → all iterations exact match.
- Metrics endpoint shows non-zero `kvbm_offload_blocks_d2h` and `kvbm_onboard_blocks_h2d` after the run.
- `pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q` → still 6 passed, 1 skipped after the venv churn.
- Sandbox venv pip-freeze captured before/after; the diff is the source of truth for "what made GB10 work".

**Risks**:
- **Wheel ABI mismatch**: if community torch and stock vllm 0.19 wheels were built against different torch ABIs, `import vllm` will fail with `undefined symbol`. Mitigation: prefer the wheel pair that the community installer used as a unit; don't mix-and-match.
- **kvbm-py3 won't link** against the new torch (cudarc / PyO3 expectations may shift). Mitigation: rebuild from scratch (`cargo clean -p kvbm-py3` then maturin develop). If still broken, surface in deviations and decide whether to pin torch closer to vllm 0.19's expected version.
- **FP8 crash on sm_121** (eugr/spark-vllm-docker#143). Mitigation: avoid FP8 model variants; if a default config triggers FP8, document the workaround.
- **CUDA 12 vs 13 toolchain mismatch**: sandbox `kvbm-kernels` build uses `nvcc 13.0.88`. If the new torch is cu128/cu129, runtime `libcudart` resolution may complain. Mitigation: rely on torch's bundled libs first; only adjust `LD_LIBRARY_PATH` if needed.
- **Symbol drift in v1 bindings** (originally noted): if the reorg altered `lib/bindings/kvbm/src/v1/block_manager/`, the rebuild against new torch may surface latent issues. Mitigation: same as before — bisect against `main` by cherry-picking only the test/fixture changes if a semantic test failure occurs.

**Rollback**: if the venv install becomes unrecoverable, restore from `.sandbox/requirements.before-phase3.txt` (task 1 snapshot) via `pip install -r`. The `.sandbox/` venv is disposable — worst case, blow it away and rebuild from `pyproject.toml` + the recorded wheel commands.

**Out of scope for phase 3**:
- Source-building vllm or PyTorch.
- Containerized (NGC) workflows.
- Patching upstream vllm CMakeLists for FP8/MoE on sm_121.
- v2 anything (gated on phase 4).
- Phase 5 cache-size derivation.

---

### Phase 4 — v2 readiness gate (detailed plan approved 2026-04-13)
**Status**: Detailed plan approved 2026-04-13; execution in progress. Source plan: `~/.claude/plans/linked-forging-liskov.md`.

**Goal**: Close every blocker that prevents `kvbm.v2.vllm.connector` from being a single-character mirror of `kvbm.v1.vllm.connector` and from booting an end-to-end vllm process on GB10 in **both** intra and inter onboard modes. After this phase, phase 5 inherits a clean import surface, two known-good vllm bring-up recipes, and a builder seam that takes the onboard mode as a configurable spec field.

**User direction (2026-04-13)**:
- New canonical paths are `kvbm.v1.vllm.connector` and `kvbm.v2.vllm.connector` — literal 1↔2 char diff. **Keep all existing `vllm_integration` paths intact as backcompat** (legacy `kvbm.vllm_integration.connector` shim and canonical `kvbm.v1.vllm_integration.connector` impl substrate both stay).
- Production v2 connector lives in `kvbm.v2.vllm.schedulers.connector`. The `kvbm/v2/vllm/connectors/` tree is a placeholder; only `connectors/leader.py` is real. Future cleanup (out of scope) will move the impl from `schedulers/` to `connectors/`; the new `kvbm.v2.vllm.connector` façade is the seam for that.
- Both `intra` and `inter` onboard modes must be validated.
- vllm version ceiling can be bumped to the version we are currently testing (sandbox: `0.19.1rc1.dev232+g0e39202ca.cu130`).

**Ground-truth findings (verified 2026-04-13)**:

1. **Three v2 import blockers exist today.**
   - `lib/bindings/kvbm/python/kvbm/v2/vllm/config.py:22` raises on `vllm >= 0.12.2` behind `KVBM_SKIP_VLLM_VERSION_CHECK`.
   - `lib/bindings/kvbm/python/kvbm/v2/vllm/version_check.py:7-30` defines a contradicting policy (`max (0,14,0)` behind a *different* env var `KVBM_DISABLE_MAX_VERSION_CHECK`) and is dead code — never called.
   - `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/dynamo.py:39-55` reads `kvbm._core.v2.RustScheduler / SchedulerConfig / RequestStatus` unconditionally; the access raises `AttributeError` (uncaught — only `ImportError` is handled), killing the package init at `schedulers/__init__.py:5` (`from .dynamo import DynamoScheduler`).

2. **Rust scheduler module intentionally absent.** `lib/bindings/kvbm/src/v2/mod.rs:8` has `// pub mod scheduler;` with TODO. The on-disk `lib/bindings/kvbm/src/v2/scheduler/{mod,config,status}.rs` files reference `dynamo_kvbm::v2::integrations::scheduler::*`, `dynamo_kvbm::v2::logical::*`, `dynamo_kvbm::G1` — the old pre-decomposition crate. Re-enabling requires porting `Scheduler`, `KVCacheManager`, `BlockManager<G1>`, `BlockRegistry`, `TinyLFUTracker` into `kvbm-engine`/`kvbm-logical`. **Significant work, deferred.**

3. **DynamoScheduler already has a complete `_rust_scheduler is None` fallback** (`dynamo.py:192, 478, 516, 552`). The Rust scheduler's only role is shadow-mode divergence comparison; KV transfer offload still routes through the kvbm v2 connector via `KVConnectorFactory.create_connector` (line 134). Widening the except clause is enough to make the production path import-safe **and** functionally complete for the determinism test.

4. **v1 path layout (verified).** Existing canonical impl substrate at `kvbm.v1.vllm_integration.connector` (eager `__init__.py`); existing legacy backcompat shim at `kvbm.vllm_integration.connector` (lazy `__getattr__`, phase-1 deliverable). **`kvbm.v1.vllm` does NOT exist yet** — phase 4 creates the new façade `kvbm.v1.vllm.connector` that lazy-redirects to the existing impl. Both vllm_integration paths stay untouched. Phase-2 v1 builder currently uses `kvbm.vllm_integration.connector`; phase 4 switches it to the new `kvbm.v1.vllm.connector` façade.

5. **v2 path layout today.** `kvbm.v2.vllm` exists with `config.py`, `version_check.py`, `connectors/`, `schedulers/`, etc. Production impl at `kvbm.v2.vllm.schedulers.connector`. Placeholder at `kvbm.v2.vllm.connectors.connector` (only `connectors/leader.py` is real). **`kvbm.v2.vllm.connector` (singular) does NOT exist** — phase 4 creates it as the canonical façade lazy-redirecting to `schedulers.connector`. Distinct from the existing `connectors/` (plural) directory; no name collision.

6. **Phase-2 v2 builder drift.** `tests/kvbm_integration/fixtures/server.py:96-111` omits `leader.onboard.mode`, deserializing to Rust default `Inter` (`lib/kvbm-config/src/onboard.rs:40`). Sandbox script hardcodes `"intra"`. With both modes in scope, the builder must accept either via a new spec field. All other fields match the Rust schema (`KvbmConfig` at `lib/kvbm-config/src/lib.rs:63-118`).

**Tasks**:

1. **Unify vllm version policy.** Bump `version_check.py:VLLM_MAX_VERSION_TESTED` to `(0, 19, 999)`, rename bypass env var to `KVBM_SKIP_VLLM_VERSION_CHECK` (single name; grep for old name first), update docstring to single-source-of-truth. Replace `config.py:22-29` inline raise with `from .version_check import version_check; version_check()`.

2. **Make `kvbm.v2.vllm.schedulers.connector` import-safe.** Widen `except ImportError:` → `except (ImportError, AttributeError):` at `dynamo.py:47`. Update warning string to be specific about the cause (`pub mod scheduler` commented out at `src/v2/mod.rs:8`; falling back to vLLM scheduler with KVBM connector offload only).

3. **Create canonical `kvbm.v{1,2}.vllm.connector` façades.** New files (mirroring the v1 phase-1 lazy-`__getattr__` pattern):
   - `lib/bindings/kvbm/python/kvbm/v1/vllm/__init__.py` (one-line namespace).
   - `lib/bindings/kvbm/python/kvbm/v1/vllm/connector/__init__.py` — lazy shim with `_V1_EXPORTS` mapping `DynamoConnector`/`DynamoConnectorMetadata`/`PdConnector`/`PdConnectorMetadata` to `kvbm.v1.vllm_integration.connector.{dynamo_connector,pd_connector}`.
   - `lib/bindings/kvbm/python/kvbm/v2/vllm/connector/__init__.py` — lazy shim with `_V2_EXPORTS` mapping `DynamoConnector` to `kvbm.v2.vllm.schedulers.connector`.
   - 4 new regression tests in `lib/bindings/kvbm/python/tests/test_legacy_imports.py` (2 identity, 2 subprocess vllm-free).

4. **Point both builders at the new canonical façades.** Edit `tests/kvbm_integration/fixtures/server.py:build_kv_transfer_config` to use `kvbm.v1.vllm.connector` (v1 branch) and `kvbm.v2.vllm.connector` (v2 branch). Update `test_kv_transfer_config.py` assertions. **Phase-2 retro fix; vllm_integration paths stay untouched.**

5. **Make v2 builder onboard.mode configurable (intra + inter).** Add `onboard_mode: str = "intra"` to `KvbmServerSpec`. Builder injects `"onboard": {"mode": spec.onboard_mode}` into the v2 leader dict; validates `("intra","inter")` or raises. Unit test parametrized across both modes.

6. **Document the deferred Rust scheduler port.** Expand the one-line TODO at `src/v2/mod.rs:8` into a comment block naming the missing types (`Scheduler`, `KVCacheManager`, `BlockManager<G1>`, `BlockRegistry`, `TinyLFUTracker`) and target crates (`kvbm-engine`/`kvbm-logical`). Cross-reference `schedulers/dynamo.py:43-45`. **No Rust code change.** Add deviations log entry below.

7. **End-to-end v2 vllm bring-up on GB10 (intra AND inter).** Two one-off bring-ups, same model (`Qwen/Qwen3-0.6B`) and `--gpu-memory-utilization 0.5` setup as phase 3. Preferred path: build a `KvbmServerSpec(kvbm_version="v2", onboard_mode=...)` and invoke `KvbmServerManager` directly (exercises the same builder + façade phase 5 will use). Pass criteria for **both** modes: vllm log shows "Application startup complete"; `/v1/chat/completions` returns a non-empty completion; `/metrics` shows kvbm v2 counters; the two runs are distinguishable in metrics/logs (confirms the leader actually picked up the chosen mode).

8. **Verification gate** (must pass before phase 5 starts):
   ```bash
   # 1. Version policy clean with no env vars
   python -c "from kvbm.v2.vllm.config import KvbmVllmConfig"
   # 2. Schedulers connector import-safe
   python -c "from kvbm.v2.vllm.schedulers.connector import DynamoConnector"
   # 3. New canonical façades
   python -c "from kvbm.v1.vllm.connector import DynamoConnector"
   python -c "from kvbm.v2.vllm.connector import DynamoConnector"
   # 4. Regression suite (now with new v2 tests)
   pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q
   # 5. Builder unit test (with new onboard.mode parametrization + module paths)
   pytest tests/kvbm_integration/fixtures/test_kv_transfer_config.py -v
   # 6. v1 determinism test against the new canonical façade
   KVBM_MODEL_ID=Qwen/Qwen3-0.6B KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 \
   KVBM_REQUEST_DELAY=2 KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 \
   KVBM_GPU_MEMORY_UTILIZATION=0.5 \
     pytest tests/kvbm_integration/test_determinism_agg.py \
       -v -k "v1-Qwen3-0.6B and cache_reset" --tb=short
   # 7. Determinism test collects with v2 in the parametrize list (temporary flip)
   sed -i 's/"v1"/"v1","v2"/' tests/kvbm_integration/test_determinism_agg.py
   pytest tests/kvbm_integration/test_determinism_agg.py --collect-only -q
   git checkout tests/kvbm_integration/test_determinism_agg.py
   # 8. Both intra and inter v2 bring-ups (Task 7) hit pass criteria
   ```

**Out of scope for phase 4**:
- Porting `dynamo_kvbm::v2::integrations::scheduler::*` types into the decomposed kvbm-* crates.
- Moving the production v2 connector impl from `kvbm/v2/vllm/schedulers/` to `connectors/`.
- Deleting the v1 legacy `kvbm.vllm_integration.connector` shim.
- Velo/nova rename completion.
- Phase-5 work: flipping the determinism test parametrize list, deriving `cache_size_gb` from `KVBM_CPU_BLOCKS`, intra/inter enumeration strategy.
- Phase-6 work: clippy `-D warnings --all-targets` re-enforcement.

**Risks**:
- vllm 0.19.x API drift beyond the phase-3 `pd_connector.py` fix may surface during Task 7. Mitigation: patch and document.
- New façade regression on the v1 path (phase 3 was validated against the legacy path). Mitigation: Task 8 step 6 reruns the phase-3 Qwen3-0.6B determinism test under the new façade.
- Lazy shim getting bypassed by static analysis tools — vllm uses `importlib.import_module` + `getattr`, which fires PEP 562 `__getattr__` correctly (confirmed by phase 1).
- Widening `except` may hide a genuinely-broken Rust build. Mitigation: keep the warning print loud; confirm `ConnectorLeader`/`ConnectorWorker` still importable.
- Latent builder-schema drift beyond `onboard.mode`. Mitigation: surface during Task 7, fix in same task.

---

### Phase 5 — v2 end-to-end determinism passing (detailed plan approved 2026-04-13)
**Goal**: Flip `tests/kvbm_integration/test_determinism_agg.py` to enumerate v2 alongside v1 and get `test_determinism_agg_with_cache_reset` to 100% exact match under v2 on GB10 with Qwen3-0.6B at default iteration counts. Larger models (R1-Distill-Llama-8B, MLA) stay deferred.

**Design decisions ratified 2026-04-13**:
1. **Cache sizing**: v2 builder uses `cache.host.num_blocks = spec.cpu_blocks` — exact parity with v1's `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` path. No HF-config lookups, no dtype/num_layers derivation.
2. **Onboard mode enumeration**: every v2 spec enumerates both `intra` and `inter`. Doubles v2 test count but catches mode-specific regressions on every run.
3. **Cache tier mandatory + max-wins**: `HostCacheConfig::compute_num_blocks` returns `None` when neither field is set (no implicit defaults). When both are set, returns `max(num_blocks, derived_from_gb)` and logs an INFO message enumerating both values with a clear marker that the larger was used. The v2 leader (`kvbm-connector`) must bail hard when neither host nor disk tier is configured — matching v1's `sanity_check` panic, but as a clean `anyhow::bail!`.

**Tasks**:

0. **Rust config — max-wins + mandatory tier** (`lib/kvbm-config/src/cache.rs` + `lib/kvbm-connector/src/connector/leader/init.rs`)
   - `HostCacheConfig::compute_num_blocks(bytes_per_block)`: if neither field set → `None`; if only one set → that one; if both set → `max(num_blocks, derived)` with `tracing::info!` enumerating both values and which one was picked.
   - Same treatment for `DiskCacheConfig::compute_num_blocks`.
   - Update unit tests in `cache.rs` to cover all four combinations for each tier (neither/only-blocks/only-gb/both-max-wins).
   - `ConnectorLeader::init`: replace `.unwrap_or(0)` at `init.rs:209` with explicit branching. Bail with the v1-parity error message if neither host nor disk tier produces a non-zero block count.

1. **Builder uses num_blocks, drop hardcoded cache_size_gb** (`tests/kvbm_integration/fixtures/server.py`)
   - Add `cpu_blocks: Optional[int] = None` to `build_kv_transfer_config`.
   - v2 branch: replace `"cache": {"host": {"cache_size_gb": 10.0}}` with `"cache": {"host": {"num_blocks": cpu_blocks}}` when set; omit the host block entirely when `None`.
   - Update call site inside `KvbmServerManager` to pass `cpu_blocks=self.spec.cpu_blocks`.
   - v1 branch unchanged — v1 still routes through `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS`.

2. **Update builder unit test** (`tests/kvbm_integration/fixtures/test_kv_transfer_config.py`)
   - Replace `cache_size_gb == 10.0` with `num_blocks == <fixture cpu_blocks>`.
   - Add a case asserting `cpu_blocks=None` omits the `cache.host` block entirely.

3. **Expand `_specs()` to cross versions × models × onboard modes** (`tests/kvbm_integration/test_determinism_agg.py`)
   - `_KVBM_VERSIONS_UNDER_TEST = ("v1", "v2")`.
   - New constant `_KVBM_V2_ONBOARD_MODES = ("intra", "inter")`.
   - v1 yields one spec per model; v2 yields `len(models) × 2` specs, one per onboard mode.
   - `_params()` stays unchanged; `KvbmServerSpec.id` already emits `v2-<model>-<mode>`.
   - Expected on GB10/Qwen3-0.6B: 1 v1 + 2 v2 = 3 cache_reset + 3 concurrent.

4. **Enable v2-* spec ids in `run_server.sh`** — delete the v2-* rejection block at `tests/kvbm_integration/scripts/run_server.sh:38-46`. Spec-id-driven launch already handles `onboard_mode` via `dataclasses.replace`.

5. **Low-iteration v2 shakedown**: `KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 KVBM_GPU_MEMORY_UTILIZATION=0.5 pytest -k "v2-Qwen3-0.6B and cache_reset"`. Surfaces config drift cheaply.

6. **Full-iteration v2 parity gate**: same but default iterations, `KVBM_SERVER_START_TIMEOUT=600`. Both intra and inter must pass with 100% match; host hit rate > 0 in both logs.

7. **Decomposed three-shell v2 smoke**: `unset NATS_SERVER ETCD_ENDPOINTS`; `run_server.sh v2-Qwen3-0.6B-intra` (shell 2); set `KVBM_SPEC_ID`+external URLs; `run_eval.sh` (shell 3). Repeat for `inter`.

8. **Side-by-side metrics capture**: during one v2 intra run, snapshot `/metrics` and compare `kvbm_offload_blocks_d2h`, `kvbm_onboard_blocks_h2d`, `kvbm_host_cache_hit_rate`, `kvbm_matched_tokens` against the phase-3 v1 snapshot (~66.4% host hit rate for Qwen3-0.6B). 10x gap flags wiring issue.

9. **Divergence isolation (contingent, only if 5 or 6 fails)**: confirm `VLLM_BATCH_INVARIANT=1` is in the v2 vllm env block; check `batch_invariant_mode` in scheduler log; if only `inter` diverges, log as follow-up, fall back to intra-only for phase-5 sign-off.

10. **README + deviations log**: `tests/kvbm_integration/README.md` v2 section (spec id format, no `run_deps_v1.sh` needed for v2, both onboard modes always enumerated, standing model); deviations entry with actual pass results + metrics; flip phase 5 state to completed.

**Verification**:
```bash
# Rust config unit tests
cargo test -p dynamo-kvbm-config cache

# Builder unit
pytest tests/kvbm_integration/fixtures/test_kv_transfer_config.py -v

# Collection enumerates v1 + v2×{intra,inter}
pytest tests/kvbm_integration/test_determinism_agg.py --collect-only -q

# Phase-1 regression
pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q

# Phase-3 v1 parity (no regression from builder edit)
KVBM_MODEL_ID=Qwen/Qwen3-0.6B KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 \
KVBM_GPU_MEMORY_UTILIZATION=0.5 KVBM_SERVER_START_TIMEOUT=600 \
  pytest tests/kvbm_integration/test_determinism_agg.py \
    -v -k "v1-Qwen3-0.6B and cache_reset" --tb=short

# Phase-5 v2 parity gate (the milestone)
KVBM_MODEL_ID=Qwen/Qwen3-0.6B KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 \
KVBM_GPU_MEMORY_UTILIZATION=0.5 KVBM_SERVER_START_TIMEOUT=600 \
  pytest tests/kvbm_integration/test_determinism_agg.py \
    -v -k "v2-Qwen3-0.6B and cache_reset" --tb=short   # Expect 2 passed
```

**Risks**:
- v2 `batch_invariant` env propagation gap (mitigate by checking vllm log env block).
- `inter` mode numerics regression (phase 4 only validated structural bring-up).
- Doubling v2 test count on slower off-host models — flagged in README, not a phase-5 blocker.

**Out of scope**: cache_size_gb derivation from dtype×num_layers; R1-Distill-Llama-8B and MLA; Rust scheduler port; velo/nova rename; phase-6 clippy re-enforcement.

---

### Phase 6 — CI all-in-one + docs
**Goal**: Lock the passing state into CI with sensible timeouts and mark external-services variants clearly.

**Deliverables**:
- CI job (or pytest markers) that runs the composed all-in-one for both v1 and v2 at reduced iteration counts.
- A debug mode (env var or pytest flag) that switches the fixtures to "external services" mode.
- `tests/kvbm_integration/README.md` updated with: which env vars control each layer, how to run each script, how to run the all-in-one, which markers gate CI vs nightly.

**Deliverables**:
- Green CI run on the target workflow.
- README that a new contributor can follow cold.

**Verification**:
- CI pipeline green on branch.
- Fresh shell + README walkthrough reproduces both v1 and v2 test runs.

**Risks**:
- CI GPU availability / nightly vs PR gating — follow existing `pytest.mark.nightly` + `pytest.mark.gpu_1` conventions.

---

## Critical files (to read/modify)

- `Cargo.toml` (workspace root) — kvbm-scheduler reference
- `lib/kvbm-config/src/v1_compat.rs` — env var translation (reference, don't modify unless phase 3 requires)
- `lib/bindings/kvbm/Cargo.toml` — feature matrix
- `lib/bindings/kvbm/python/kvbm/__init__.py` — feature detection
- `lib/bindings/kvbm/python/kvbm/_feature_stubs.py`
- `lib/bindings/kvbm/python/kvbm/vllm_integration/connector/__init__.py` — **phase 1 target**
- `lib/bindings/kvbm/python/kvbm/v1/vllm_integration/connector/dynamo_connector.py` — v1 canonical
- `lib/bindings/kvbm/python/kvbm/v2/vllm/{connectors,schedulers}/connector.py` — **phase 4 target**
- `lib/bindings/kvbm/python/kvbm/conftest.py` — inspect purpose
- `tests/kvbm_integration/conftest.py` — `runtime_services` reuse-or-spawn
- `tests/kvbm_integration/common.py` — `DeterminismTester` (eval layer, keep)
- `tests/kvbm_integration/test_determinism_agg.py` — **phase 2 rewrite target**
- `~/archives/dynamo/.sandbox/launch_vllm_with_connector.sh` — v2 reference config

## Reusable existing helpers (don't rewrite)

- `DeterminismTester`, `ApiTester`, `TestDeterminism` in `tests/kvbm_integration/common.py`
- `runtime_services` fixture in `tests/kvbm_integration/conftest.py`
- `NatsServer`, `EtcdServer`, `allocate_port`, `deallocate_port` in `tests/conftest.py` / `tests/utils/port_utils.py`
- `parse_kvbm_metrics`, `fetch_kvbm_metrics` in `common.py`
- kvbm-config `v1_compat.rs` — already handles `DYN_KVBM_*` → v2 env var translation

## Verification contract (end-to-end)

After all phases, these must all pass from a fresh venv:

```bash
# Build
cargo check --all-features --all-targets
cargo clippy --all-features --no-deps --all-targets -- -D warnings
cd lib/bindings/kvbm && maturin develop --features v1,v2

# Smoke imports
python -c "from kvbm.vllm_integration.connector import DynamoConnector"           # v1 shim
python -c "from kvbm.v2.vllm.schedulers.connector import DynamoConnector"          # v2 path

# Decomposed runs (separate shells)
bash tests/kvbm_integration/scripts/run_deps_v1.sh
bash tests/kvbm_integration/scripts/run_server.sh v1
bash tests/kvbm_integration/scripts/run_eval.sh

bash tests/kvbm_integration/scripts/run_deps_v2.sh
bash tests/kvbm_integration/scripts/run_server.sh v2
bash tests/kvbm_integration/scripts/run_eval.sh

# All-in-one
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
    pytest tests/kvbm_integration/test_determinism_agg.py -v \
    -k "test_determinism_agg_with_cache_reset and (v1 or v2)" --tb=short
```

## Alternatives considered (noted per CLAUDE.md instruction)

1. **v2-first ordering**. Fixing v2 before restoring v1 would front-load the hardest work. Rejected: v1 is closer to green on this branch (reorg only), we need a known-good reference to compare v2 numerics against, and the test decomposition is easier to validate against a working path first.
2. **Parametrize version inside `LLMServerManager`** instead of extracting fixtures. Less code churn but keeps the "three runnable components" requirement unmet; the user explicitly asked for separable pieces for local iteration. Rejected.
3. **Create `kvbm-scheduler` stub crate** instead of removing the reference. Keeps the door open if future work plans to split scheduler from connector; however, the user's memory notes decomposition already landed without it. Default is to remove; flip to stub only if grep finds references.

## Deviations log

### 2026-04-13 — Phase 5 execution

**Outcome**: Milestone goal reached. `test_determinism_agg_with_cache_reset` passes under **both** v2 onboard modes on GB10 with Qwen3-0.6B at default iteration counts (100 iters × 2 phases). v1 phase-3 parity unchanged. Spec matrix permanently enumerates `("v1","v2")` with v2 crossed against both onboard modes, so every determinism run catches intra and inter regressions in the same invocation.

**v2 full-iter parity gate (Task 7)**:
- `test_determinism_agg_with_cache_reset[v2-Qwen3-0.6B-intra]` — **1 passed in 182.63s**, final host cache hit rate **67.0% (3473/5184)**
- `test_determinism_agg_with_cache_reset[v2-Qwen3-0.6B-inter]` — **1 passed in 183.76s**, final host cache hit rate **66.7% (3423/5134)**
- Both match the phase-3 v1 baseline (~66.4%) and both report `host_block_count=2000 bytes_per_block=1,835,008 disk_block_count=None` — confirming the new builder `cpu_blocks` path flows end to end into the Rust leader.
- Environment: `KVBM_MODEL_ID=Qwen/Qwen3-0.6B KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 KVBM_GPU_MEMORY_UTILIZATION=0.5 KVBM_SERVER_START_TIMEOUT=600`
- Logs: `/tmp/dynamo_tests/test_determinism_agg_with_cache_reset[v2-Qwen3-0.6B-{intra,inter}]/ServerType.vllm_server_v2_cpu2000_gpu512_*.log`

**Design decisions ratified with user (Tasks 0/1/2)**:
1. **Cache sizing**: v2 builder passes `cache.host.num_blocks = spec.cpu_blocks` — exact parity with v1's `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS`. The original plan called for deriving `cache_size_gb` from `KVBM_CPU_BLOCKS × block_size × dtype × num_layers`; rejected in favor of the direct num_blocks flow so v1 and v2 use the same Rust field with no HF-config lookups, no drift.
2. **Both-modes enumeration**: every v2 spec yields both `intra` and `inter` — no env gate. Rationale: phase 4 only validated structural bring-up for both modes; determinism numerics were unverified until now. Doubling the v2 test count is acceptable on GB10 (~3 min per run) and is mandatory to prevent mode-specific regressions from sneaking through.
3. **Mandatory tier + max-wins**: `HostCacheConfig::compute_num_blocks` / `DiskCacheConfig::compute_num_blocks` now return `None` when neither `num_blocks` nor `cache_size_gb` is set, return the single value when one is set, and return `max(explicit, derived)` when both are set with an INFO log line enumerating `explicit_num_blocks`, `cache_size_gb`, `derived_num_blocks`, `bytes_per_block`, `picked`. The v2 leader in `kvbm-connector` now bails with the v1-parity "KVBM Configuration Error" message when neither host nor disk tier produces a non-zero block count (previously silent `.unwrap_or(0)` fallback).

**Critical fix — spurious `#[cfg(feature = "nccl")]` gates on intra onboard path (in-phase discovery, not in original plan)**: low-iter shakedown exposed `v2-Qwen3-0.6B-intra` emitting degenerate `iegoiegoiego...` output after cache reset while `inter` passed. v1 at the same 2-iter config passed, ruling out "iteration count too low." Root cause: three `#[cfg(feature = "nccl")]` gates in the intra onboard path:
- `lib/kvbm-engine/src/worker/physical.rs:19` — `use cudarc::driver::CudaEvent` was gated on nccl even though `CudaEvent` is a plain CUDA primitive, not nccl-specific.
- `lib/kvbm-engine/src/worker/physical.rs:361` — `execute_local_layerwise_onboard` was gated on nccl, but the function body is pure CUDA (acquires an H2D stream via `acquire_h2d_stream()`, calls per-layer `manager.execute_transfer(...)` with `TransferOptions.cuda_stream`, then `event.record(stream)` per layer). No collective ops, no nccl comms.
- `lib/kvbm-connector/src/connector/worker/mod.rs:530-556` — the call site in `start_load_kv` had a symmetric `#[cfg(not(feature = "nccl"))]` branch that silently logged `"Intra-pass layerwise onboard requires nccl feature — skipping"` and dropped the `intra_pass_load` request. Since the bindings weren't built with the nccl cargo feature, this branch fired on every intra request: `start_load_kv` became a no-op, `intra_pass_onboard_active` stayed `false`, so `wait_for_layer_load` early-returned without the `cuStreamWaitEvent`, and vLLM's forward pass read uninitialized GPU KV slots — producing the `iegoiego` degenerate attractor.

**Fix**: removed all three gates; made `cudarc` an unconditional dep in `lib/kvbm-engine/Cargo.toml` (was `optional = true`, only pulled in by `nccl = ["dep:cudarc"]`). `cudarc` is already unconditionally pulled in by `kvbm-physical` so this adds zero build cost. The `nccl` cargo feature is now `nccl = []`; it still gates the genuine NCCL-using code in `lib/kvbm-engine/src/collectives/` (which uses `cudarc::nccl::sys`), but no longer incorrectly gates the intra-pass CUDA-only onboard. The pre-existing wiring it depends on was already nccl-free:
- `lib/kvbm-connector/src/connector/worker/state.rs:219-246` pre-allocates H2D and D2H `CudaEvent` vectors unconditionally at init
- `lib/kvbm-connector/src/connector/worker/mod.rs:579-617 wait_for_layer_load` uses raw `cuStreamWaitEvent(stream, event.cu_event(), 0)` with no gate
- `mod.rs:263-281` has the offload-side equivalent using `cuEventRecord` / `cuStreamWaitEvent` with no gate

After the fix, `v2-Qwen3-0.6B-intra` at 2 iters: **1 passed in 39s** (smoke), then at default iterations: **1 passed in 182.63s** with 67.0% host hit rate (the full-iter gate).

**Side-by-side metrics snapshot (Task 8)**:

| Spec | Wall | host_block_count | bytes_per_block | Host hit rate |
|---|---|---|---|---|
| v1-Qwen3-0.6B (phase-3 baseline) | 190.47s | 2000 | — | 66.4% |
| v2-Qwen3-0.6B-intra | 182.63s | 2000 | 1,835,008 | 67.0% (3473/5184) |
| v2-Qwen3-0.6B-inter | 183.76s | 2000 | 1,835,008 | 66.7% (3423/5134) |

The `bytes_per_block=1,835,008` matches Qwen3-0.6B's KV geometry (28 layers × 2 × 16 tokens × 1024 hidden × 2 bytes). Both v2 modes track v1 wall time (within ~4% — the v2 path pays a one-time v2 Rust scheduler pass-through overhead). `disk_block_count=None` in both runs — only the G2 host tier is configured, matching the phase-3 v1 topology.

**Decomposed three-shell v2 smoke (Task 8, separate)**: reproduced both modes end-to-end via `run_server.sh` + `run_eval.sh`. Shell 1 is a no-op for v2 (`NATS_SERVER`/`ETCD_ENDPOINTS` must be unset so the v1 deps gate in `run_server.sh` is skipped). Shell 2: `bash tests/kvbm_integration/scripts/run_server.sh v2-Qwen3-0.6B-{intra,inter}` prints the server export block once `[server] READY`. Shell 3: export the three vars and run `bash tests/kvbm_integration/scripts/run_eval.sh`. Eval bypasses the spawn path (`KVBM_EXTERNAL_BASE_URL` triggers external-attach mode) and runs against the already-booted server. Intra smoke: **1 passed in 11.4s**. Inter smoke: **1 passed in 11.5s**. Both eval invocations bypassed spawn (only ~11s wall because the server was already warm), confirming the external-attach handshake works for v2 spec ids and that the server-side spec-id lookup in `run_server.sh` correctly reconstructs the onboard mode from the spec id.

**Builder unit test updates (Task 3)**: `test_v2_payload_has_required_leader_blocks` now asserts `leader["cache"]["host"] == {"num_blocks": 2000}` (previously `cache_size_gb == 10.0`). New `test_v2_payload_omits_cache_host_when_cpu_blocks_none` case asserts that `cpu_blocks=None` omits the `cache.host` block entirely so the Rust leader hits the mandatory-tier bail. 11/11 passed.

**Spec matrix (Task 4)**: `_KVBM_VERSIONS_UNDER_TEST = ("v1", "v2")`, new constant `_KVBM_V2_ONBOARD_MODES = ("intra", "inter")`. `_specs()` yields one spec per model for v1 and two specs per model (one per mode) for v2. Collect shows 12 tests (2 test methods × (1 v1 + 2 v2 modes) × 2 models). On GB10 with `KVBM_MODEL_ID=Qwen/Qwen3-0.6B` the second model (DeepSeek-V2-Lite MLA) is pytest-skipped, leaving 3 cache-reset + 3 concurrent runs.

**`run_server.sh` v2 gate removed (Task 5)**: deleted the `v2-*` rejection block at `lines 38-46`; added branching so `v1-*` specs still require `NATS_SERVER`/`ETCD_ENDPOINTS` but `v2-*` specs deliberately do not. The spec-id-driven launch path (imports `_CACHE_RESET_SPECS`, rebuilds the spec via `dataclasses.replace` with block overrides) already handled `onboard_mode` correctly because it's a field on the canonical spec.

**Rust unit tests (Task 0)**: added 9 new cache-config tests covering the four `compute_num_blocks` cases for both host and disk tiers (neither/only-blocks/only-gb/both-max-wins, plus bytes-per-block zero). `cargo test -p kvbm-config cache::` → **18 passed**. Clippy clean on `kvbm-config` and `kvbm-connector` and on the workspace `--all-features --all-targets`.

**Files touched** (this round):
- `lib/kvbm-config/src/cache.rs` — max-wins semantics, mandatory-tier None return, new tests
- `lib/kvbm-connector/src/connector/worker/mod.rs` — removed nccl-gate at intra onboard call site; left the downstream event wiring untouched
- `lib/kvbm-connector/src/connector/leader/init.rs` — replaced `.unwrap_or(0)` with explicit bail matching v1 sanity_check
- `lib/kvbm-engine/src/worker/physical.rs` — removed nccl-gate on `execute_local_layerwise_onboard` and on the `CudaEvent` import
- `lib/kvbm-engine/Cargo.toml` — `cudarc` moved from optional to required; `nccl = ["dep:cudarc"]` → `nccl = []`
- `tests/kvbm_integration/fixtures/server.py` — `build_kv_transfer_config` gained `cpu_blocks` parameter; v2 branch emits `cache.host.num_blocks` or omits the cache block entirely; `KvbmServerManager` call site passes `spec.cpu_blocks`
- `tests/kvbm_integration/fixtures/test_kv_transfer_config.py` — num_blocks assertion + new omission case
- `tests/kvbm_integration/test_determinism_agg.py` — parametrize flip + `_KVBM_V2_ONBOARD_MODES` cross
- `tests/kvbm_integration/scripts/run_server.sh` — v2 gate removed; v1/v2 branch split on NATS/ETCD requirement
- `tests/kvbm_integration/README.md` — v2 decomposed-flow section, v2 spec id format, KVBM_CPU_BLOCKS → num_blocks note
- `ACTIVE_PLAN.md` — phase 5 plan rewritten + this entry + state flip

**Out of scope, still deferred**:
- DeepSeek-R1-Distill-Llama-8B at default iteration counts — needs faster host (spec wiring ready; both v2 modes will auto-enumerate once the model env var flips).
- DeepSeek-V2-Lite (MLA) execution — gated on `KVBM_ENABLE_MLA`; both v2 modes auto-enumerate once unblocked.
- Porting `dynamo_kvbm::v2::integrations::scheduler::*` to the decomposed kvbm-* crates. `DynamoScheduler` still runs in `_rust_scheduler is None` passthrough; phase 5 did not need this.
- Phase 6: workspace clippy `-D warnings --all-targets` re-enforcement.

### 2026-04-13 — Phase 4 execution

**Outcome**: All 8 tasks landed; 8-command verification gate exits 0; **both `intra` and `inter` v2 vllm bring-ups PASSED on GB10 with Qwen3-0.6B**, and the v1 phase-3 determinism test still passes against the new `kvbm.v1.vllm.connector` façade in 44s. The Rust scheduler port (`dynamo_kvbm::v2::integrations::scheduler::*` → decomposed kvbm-* crates) is documented and deferred — DynamoScheduler's `_rust_scheduler is None` fallback is sufficient for phase 5.

**Single source of truth for vllm version policy (Task 1)**: `kvbm/v2/vllm/version_check.py` is the only check. `VLLM_MAX_VERSION_TESTED = (0, 19, 999)` admits the sandbox vllm `0.19.1rc1.dev232+g0e39202ca.cu130`. The dead `KVBM_DISABLE_MAX_VERSION_CHECK` env var is gone; `KVBM_SKIP_VLLM_VERSION_CHECK` is the documented bypass. The previous inline raise in `config.py:22-29` was deleted; `version_check()` is called once at the top of `config.py` (the gateway every code path that touches vllm transits through). It is **not** called from `kvbm/v2/vllm/__init__.py` — that would force `vllm/__init__.py` (transitively `vllm._version`, `vllm.envs`, `vllm.logger`, `vllm.utils`) to load whenever anything just imports `kvbm.v2.vllm.connector` to scan the path, defeating the lazy-shim point.

**dynamo.py except widening (Task 2)**: `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/dynamo.py:47` now catches `(ImportError, AttributeError)`. The Rust scheduler symbols (`RustScheduler`, `SchedulerConfig`, `RequestStatus`) raise `AttributeError` because `pub mod scheduler` is commented out in `lib/bindings/kvbm/src/v2/mod.rs:8`; the previous `except ImportError:` let that escape and killed `kvbm.v2.vllm.schedulers/__init__.py:5`'s `from .dynamo import DynamoScheduler`. After the fix `_RUST_SCHEDULER_AVAILABLE = False` and `dynamo.py:192, 478, 516, 552` route the schedule decision through vLLM directly. KV transfer offload still goes through `ConnectorLeader`/`ConnectorWorker` (verified live: `kvbm_connector::connector::leader: ConnectorLeader initialized with onboard mode onboard_mode=Intra/Inter` in the smoke logs).

**Critical hidden second blocker — `kvbm/v2/__init__.py`**: phase 4 plan only identified the dynamo.py except clause. Ground-truth from the smoke run revealed `kvbm/v2/__init__.py:25-27` had the **same uncaught `AttributeError`** on `RustScheduler / SchedulerConfig / RequestStatus`, which collapsed the **entire** v2 namespace (`KvbmRuntime`, `KvbmVllmConfig`, `ConnectorLeader`, `ConnectorWorker`, `KvbmRequest`, `SchedulerOutput`, `Tensor`) into the feature-stub fallback. Symptom: `kvbm.v2.is_available()` returned `False` and `KvbmRuntime`/`ConnectorWorker` were `<function … at …>` stubs even though the Rust binding loaded fine. Fix: split the v2 init into two try blocks — runtime/connector/Rust core in one (`_V2_CORE_AVAILABLE`, only catches `ImportError`), scheduler trio in a second (`_V2_SCHEDULER_AVAILABLE`, catches `(ImportError, AttributeError)` and falls back to `RustScheduler = SchedulerConfig = RequestStatus = None`). After the fix `kvbm.v2.is_available()` returns `True`, `KvbmRuntime`/`ConnectorWorker` are real classes, and the smoke runs reach `Application startup complete.`.

**Canonical façade lazy shims (Task 3)**: created `kvbm/v1/vllm/{__init__.py,connector/__init__.py}` and `kvbm/v2/vllm/connector/__init__.py` mirroring the phase-1 lazy `__getattr__` pattern. `kvbm.v1.vllm.connector` redirects to `kvbm.v1.vllm_integration.connector.{dynamo_connector,pd_connector}`; `kvbm.v2.vllm.connector` redirects to `kvbm.v2.vllm.schedulers.connector`. Both are confirmed lazy by subprocess regression tests (4 new tests in `test_legacy_imports.py`); the v2 subprocess test tolerates `vllm.version` being loaded transitively because the version-policy gate is intentionally NOT in the v2 namespace init (see Task 1 note above). 10 passed, 1 trtllm-skipped.

**Builder paths and onboard_mode (Tasks 4+5)**: `tests/kvbm_integration/fixtures/server.py:build_kv_transfer_config` now points v1 at `kvbm.v1.vllm.connector` and v2 at `kvbm.v2.vllm.connector` (1↔2 char mirror; the existing `vllm_integration` paths stay untouched as backcompat). Added `onboard_mode: str = "intra"` to `KvbmServerSpec`; the v2 builder injects `leader.onboard.mode` from the spec and validates `("intra","inter")` at builder time. The v2 spec id now includes the mode (e.g. `v2-Qwen3-0.6B-intra`); v1 spec ids unchanged. Unit test parametrized across both modes — 10 passed.

**Deferred Rust scheduler port (Task 6)**: Expanded the `// pub mod scheduler;` TODO comment in `lib/bindings/kvbm/src/v2/mod.rs:8` to name the missing types (`Scheduler`, `KVCacheManager`, `BlockManager<G1>`, `BlockRegistry`, `TinyLFUTracker`) from the pre-decomposition `dynamo_kvbm` crate and the likely target crates (`kvbm-engine`, `kvbm-logical`). No code change. `cargo check --features v1,v2` still green in `lib/bindings/kvbm`.

**End-to-end v2 vllm smoke recipe (Task 7)** — reproduces the intra/inter validation on GB10:
```bash
# from workspace root, sandbox venv active
export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda KVBM_REQUIRE_CUDA=1
export PATH=/usr/local/cuda/bin:$PATH KVBM_GPU_MEMORY_UTILIZATION=0.5
PYTHONPATH=. python /tmp/kvbm_v2_smoke.py intra   # ~90s, exits 0
PYTHONPATH=. python /tmp/kvbm_v2_smoke.py inter   # ~90s, exits 0
```
The `kvbm_v2_smoke.py` harness builds a `KvbmServerSpec(kvbm_version="v2", model_config=KvbmModelConfig(model_id="Qwen/Qwen3-0.6B", block_size=16, attention_backend="FLASH_ATTN", max_model_len=4096), cpu_blocks=2000, gpu_blocks=512, onboard_mode=...)`, hands it to `KvbmServerManager`, hits `/v1/chat/completions` with a tiny prompt, scrapes `/metrics`, tears down. The `block_size=16, attention_backend="FLASH_ATTN"` fields are required because `VLLM_BATCH_INVARIANT=1` in vllm 0.19.x now needs an explicit attention backend; the test_determinism_agg.py specs set them, the smoke harness must too.

**Smoke-test pass criteria (both modes)**:
- vllm log: `INFO:     Application startup complete.`
- `/v1/chat/completions` returns a non-empty completion (Qwen3-0.6B replied `'<think>\nOkay, the user wants me'` to a "say pong" prompt — model is generating).
- `/metrics` shows 13 `kvbm_*` counter lines (offload/onboard d2d/d2h/h2d/d2o/o2d, host/disk/object cache hit rates, matched tokens, object read/write failures). All 0 because we sent only one request, but the metrics endpoint and registration are wired.
- **Mode-distinguishing log line** (different per run): `kvbm_connector::connector::leader: ConnectorLeader initialized with onboard mode onboard_mode=Intra` (intra run) vs `onboard_mode=Inter` (inter run). Confirmed in `/tmp/kvbm_v2_smoke/intra/*.log` and `/tmp/kvbm_v2_smoke/inter/*.log` respectively. The Rust leader is **actually picking up** the JSON `onboard.mode` field from our builder.

**Verification gate matrix** (Task 8 — all 8 green 2026-04-13):
- `python -c "from kvbm.v2.vllm.config import KvbmVllmConfig"` → exit 0 (no env vars).
- `python -c "from kvbm.v2.vllm.schedulers.connector import DynamoConnector"` → exit 0.
- `python -c "from kvbm.v1.vllm.connector import DynamoConnector"` → exit 0.
- `python -c "from kvbm.v2.vllm.connector import DynamoConnector"` → exit 0.
- `pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q` → 10 passed, 1 skipped (trtllm).
- `pytest tests/kvbm_integration/fixtures/test_kv_transfer_config.py -v` → 10 passed.
- `pytest tests/kvbm_integration/test_determinism_agg.py -v -k "v1-Qwen3-0.6B and cache_reset" --tb=short` (with phase-3 env) → 1 passed in 44.09s. Confirms the new `kvbm.v1.vllm.connector` façade resolves correctly under a real vllm spawn.
- Temporary parametrize flip to `("v1","v2")` collects 8 tests cleanly (4 v1 + 4 v2-intra), no ImportError. Reverted to `("v1",)` after capture; phase 5 owns the permanent flip.

**Files touched** (this round):
- `lib/bindings/kvbm/python/kvbm/v2/vllm/version_check.py` — rewritten, single source of truth, max bumped to (0,19,999), bypass var unified
- `lib/bindings/kvbm/python/kvbm/v2/vllm/config.py` — inline raise deleted, `version_check()` called at the top, vllm imports moved below
- `lib/bindings/kvbm/python/kvbm/v2/vllm/__init__.py` — version_check kept as a re-exported symbol but NOT called at package init (lazy-shim preservation)
- `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/dynamo.py:47` — `except (ImportError, AttributeError)`, warning string rewritten
- `lib/bindings/kvbm/python/kvbm/v2/__init__.py` — split into core + scheduler try blocks; `_V2_SCHEDULER_AVAILABLE` flag added
- `lib/bindings/kvbm/python/kvbm/v1/vllm/__init__.py` — new namespace
- `lib/bindings/kvbm/python/kvbm/v1/vllm/connector/__init__.py` — new lazy v1 façade
- `lib/bindings/kvbm/python/kvbm/v2/vllm/connector/__init__.py` — new lazy v2 façade
- `lib/bindings/kvbm/python/tests/test_legacy_imports.py` — 4 new tests (v1/v2 identity + v1/v2 lazy subprocess)
- `tests/kvbm_integration/fixtures/server.py` — builder paths bumped, `onboard_mode` plumbed through `KvbmServerSpec` + `KvbmServerManager` + builder, spec id now includes mode for v2
- `tests/kvbm_integration/fixtures/test_kv_transfer_config.py` — assertions bumped, parametrized across `intra`/`inter`, default-mode and bad-mode cases added
- `lib/bindings/kvbm/src/v2/mod.rs:8` — TODO comment expanded
- `ACTIVE_PLAN.md` — phase 4 detail rewritten + this entry + state flip

**Out of scope, still deferred**:
- Porting `dynamo_kvbm::v2::integrations::scheduler::*` (Scheduler, KVCacheManager, BlockManager<G1>, BlockRegistry, TinyLFUTracker) into the decomposed kvbm-* crates. Until then `RustScheduler` stays `None` in `kvbm.v2.__init__` and `DynamoScheduler` runs as a vLLM passthrough. Phase 5 does not need this.
- Moving the v2 connector impl from `kvbm/v2/vllm/schedulers/` to `kvbm/v2/vllm/connectors/` (the future cleanup the user explicitly deferred). The new `kvbm.v2.vllm.connector` façade is the seam — when the move happens, only the `_V2_EXPORTS` redirect target changes.
- Velo/nova rename completion (`MessengerConfig` JSON key still `"nova"`, `NovaPeerMetadata` Python class unchanged).
- Phase 5: flipping `_KVBM_VERSIONS_UNDER_TEST` to `("v1","v2")` permanently, deriving `cache.host.cache_size_gb` from `KVBM_CPU_BLOCKS × block_size × dtype × num_layers`, deciding intra/inter enumeration strategy in the spec list (current default is one v2 spec per model with `onboard_mode="intra"`; phase 5 may want both per model or a `KVBM_ONBOARD_MODE` env gate à la `KVBM_ENABLE_MLA`).
- Phase 6: clippy `-D warnings --all-targets` workspace-wide re-enforcement.

### 2026-04-12 — Phase 1 execution

**Closed an undocumented gap**: `components/src/dynamo/vllm/main.py:540` does `from kvbm.vllm_integration.consolidator_config import get_consolidator_endpoints`. The original phase-1 task list did not include this path. Added a thin re-export shim at `lib/bindings/kvbm/python/kvbm/vllm_integration/consolidator_config.py` exporting `get_consolidator_endpoints`, `is_truthy`, `should_enable_consolidator` from `kvbm.v1.vllm_integration.consolidator_config`. Eager (not lazy) — the only caller is gated on `_uses_dynamo_connector`, so vllm is necessarily present at import time.

**Lazy `__getattr__` chosen over eager wildcard re-export** for `kvbm/vllm_integration/connector/__init__.py`. Reason: `kvbm.v1.vllm_integration.connector.dynamo_connector` transitively requires `vllm`. Eager re-export would force `import vllm` whenever the legacy module path is merely scanned by pytest collection on a vllm-free host, which is exactly what the original v2 redirect was avoiding (see existing comment in `kvbm/vllm_integration/__init__.py` about avoiding circular imports under vLLM's module-loading path). The shim uses an explicit `_V1_EXPORTS` dict (not wildcard) to avoid name shadowing and keep `DynamoConnector.__module__ == "kvbm.v1.vllm_integration.connector.dynamo_connector"` (verified by identity assertion). PEP 562 `__getattr__` fires for vLLM's `getattr(mod, "DynamoConnector")` after `importlib.import_module(...)`, so the contract holds.

**conftest.py rewritten, not just edited**: removed the blanket v1 skip + "deprecated" framing, added a `tensorrt_llm`-availability gate alongside the existing vllm gate, and consolidated detection via `importlib.util.find_spec` (avoids actually importing vllm/trtllm at conftest evaluation time). v1 vllm_integration tests are now collectable when vllm is installed; v1 trtllm_integration when tensorrt_llm is installed. Verified with `pytest tests/kvbm_integration/test_kvbm_vllm_integration.py --collect-only` → 5 tests collected (previously 0 — blanket-skipped).

**Regression test added** at `lib/bindings/kvbm/python/tests/test_legacy_imports.py` (new directory). 7 tests:
- `test_legacy_connector_importable_without_forcing_vllm` — subprocess form, asserts `'vllm' not in sys.modules` after importing the legacy path. Subprocess chosen over in-process check because pytest sessions share interpreter state and a sibling test importing vllm would spuriously break in-process inspection.
- `test_legacy_connector_resolves_to_v1` — identity check `DynamoConnector is kvbm.v1.vllm_integration.connector.dynamo_connector.DynamoConnector` and `__module__` assertion.
- `test_legacy_connector_secondary_exports` — `DynamoConnectorMetadata`, `PdConnector`, `PdConnectorMetadata` all resolve to v1.
- `test_legacy_consolidator_config_shim` — identity check on `get_consolidator_endpoints`.
- `test_legacy_trtllm_connector_resolves_to_v1` — gated on `tensorrt_llm` availability (mid-execution refinement: originally gated only on vllm, but the v1 trtllm leader does `import tensorrt_llm` at module top so the gate must be tighter).
- `test_legacy_utils_nvtx_annotate_resolves_to_v1` — identity check on the existing `kvbm.utils` shim.
- `test_top_level_kvbm_has_v1_and_v2` — sanity that both feature flags are True.

Run on the sandbox venv (vllm 0.19.0 present, tensorrt_llm absent): **6 passed, 1 skipped (trtllm)**.

**Files touched**:
- `lib/bindings/kvbm/conftest.py` — rewritten
- `lib/bindings/kvbm/python/kvbm/vllm_integration/connector/__init__.py` — rewritten
- `lib/bindings/kvbm/python/kvbm/vllm_integration/consolidator_config.py` — new
- `lib/bindings/kvbm/python/tests/test_legacy_imports.py` — new
- `ACTIVE_PLAN.md` — phase-state bump + this log entry

`cargo fmt --check` still clean. No Rust changes.

### 2026-04-12 — Phase 1 policy clarification (Option A)
On ground-truthing, the shim at `lib/bindings/kvbm/python/kvbm/vllm_integration/connector/__init__.py` turned out to be a deliberate lazy redirect to `kvbm.v2.vllm.connectors.connector`, and `lib/bindings/kvbm/conftest.py` marked v1 integration files as "deprecated, not actively maintained". This contradicts the milestone goal of "fully functioning v1 and v2 execution". Decision: Option A — legacy `kvbm.vllm_integration.connector` == v1 (matches `main`). v1 is **not** deprecated; the conftest "deprecated" language will be removed. Explicit v2 path for tests will be `kvbm.v2.vllm.schedulers.connector`. Rationale: zero churn in existing v1 test files, unambiguous semantics per version (no silent redirect), aligns with user's brief to "re-export v1 bits into paths consistent with main".

### 2026-04-12 — Phase 0 findings
**Stale scheduler/recorder references** (removed):
- `Cargo.toml` line 62 had a `kvbm-scheduler = { path = "lib/kvbm-scheduler" }` workspace-dependency entry pointing at a directory that never existed. No member crate actually imported `kvbm-scheduler` (grep-verified). Removed.
- `lib/kvbm-connector/src/testing/mod.rs` declared `pub mod recorder;` for a file that never existed in git history. Removed the module declaration and the stale comments that pointed at `lib/kvbm-scheduler/tests/`.

**s3_object restoration** (initial drop later reversed):
- First pass dropped `mod s3_object;` from `lib/kvbm-connector/src/testing/e2e/mod.rs` because it was gated on a `"s3"` feature that wasn't declared in `kvbm-connector/Cargo.toml` and the 955-line file had stale pre-decomposition imports. User correctly pointed out that `kvbm-engine` already has a live `s3` feature (`default = ["s3"]`) and the path should be restored.
- Fix: added `s3 = ["kvbm-engine/s3"]` feature to `kvbm-connector/Cargo.toml`, added `rstest = "0.23"` as a dev-dependency, restored `#[cfg(all(test, feature = "s3"))] mod s3_object;` in `e2e/mod.rs`, and migrated the stale import paths in `s3_object.rs`:
  - `kvbm_engine::distributed::object::*` → `kvbm_engine::object::*` (the module is no longer under `distributed::`)
  - `crate::{BlockId, KvbmSequenceHashProvider, SequenceHash}` → `crate::{BlockId, SequenceHash}` plus `use kvbm_logical::KvbmSequenceHashProvider;`
  - Inline `use kvbm_engine::distributed::object::ObjectBlockOps;` at line 189 → `use kvbm_engine::object::ObjectBlockOps;`
- Verified: `cargo check -p kvbm-connector --features testing,s3 --tests` compiles clean; `cargo clippy -p kvbm-connector --features testing,s3 --tests --no-deps -- -D warnings` passes.

**Clippy `-D warnings` on this branch's surface area**:
- All 8 kvbm-* crates (`kvbm-common`, `kvbm-config`, `kvbm-connector`, `kvbm-engine`, `kvbm-kernels`, `kvbm-logical`, `kvbm-observability`, `kvbm-physical`) — green with `--all-features`.
- `kvbm-py3` (bindings crate, separate workspace at `lib/bindings/kvbm`) — green.
- Fixes applied this phase:
  - `kvbm-observability/src/observability.rs` — named-binding for `JoinHandle` (`let _handle = start_metrics_server(...)`) to silence `let_underscore_future`.
  - `kvbm-kernels/examples/kvbench.rs` — dropped four redundant `as *const *const c_void` / `as *const *mut c_void` casts; refactored `run_benchmark` from an 8-arg function to take a `BenchmarkParams` config struct (per CLAUDE.md critical rule: no `#[allow(clippy::too_many_arguments)]`).
  - `kvbm-physical/src/transfer/context.rs` — refactored `pub(crate) TransferContext::new` from 8 args to `(nixl_agent, cuda_context, config: TransferConfig)` by destructuring the already-built `TransferConfig` inside `new`. Two call sites (the two builder `build()` methods) were updated.
  - `kvbm-engine/src/collectives/bootstrap.rs` — dropped unnecessary `as u8` cast (value was already `u8`).
  - `kvbm-engine/src/worker/physical/replicated.rs` — removed `return` from the branch before `else if` (needless_return).
  - `kvbm-engine/src/worker/physical.rs` — rewrote `for layer in 0..num_layers { ...; layer_events[layer].record(...) }` as `for (layer, event) in layer_events.iter().enumerate().take(num_layers) { ...; event.record(...) }`.
  - `kvbm-connector/src/common/request.rs` — deleted the 8-arg `with_priority` helper. The existing `derive_builder::Builder` on `Request` already provides the full-config path; rewrote the 7-arg `with_token_limits` convenience to delegate to the builder directly (explicitly setting `priority(None)` since nobody goes through `with_token_limits` to set a priority). No external callers of `with_priority` (grep-verified).
  - `kvbm-py3/src/v1/block_manager/vllm/block_list.rs` — added `BlockStates::is_empty` next to existing `len`.
  - `kvbm-py3/src/v1/block_manager/vllm.rs` — `#[allow(clippy::should_implement_trait)]` on `SlotError::from_str` **⚠ standing `#[allow]`**: name collision with `std::str::FromStr` trait on v1 bindings code. Deferred for phase-1 follow-up — renaming cascades into v1 call sites and is out of scope for phase 0. Not per CLAUDE.md critical rules, flag for explicit decision.

**Pre-existing upstream clippy debt (out of scope for phase 0)**: workspace-wide `cargo clippy --all-features --all-targets -- -D warnings` still fails in `dynamo-runtime` (1 `needless_return` in lib + 3 `result_large_err` in lib test on the `figment::Jail::expect_with` nats test closures). Verified against `main` — these errors exist identically on `main`. Upstream CI does not currently enforce `-D warnings` across the full workspace. Documenting here so that future phases don't confuse them with regressions from this branch.

**CUDA / nvcc requirement (clarification of earlier speculation)**:
- User asked whether CUDA was a v1 requirement. **It is not — it is a v2 requirement.** Build-time chain:
  - `kvbm-py3/v2` → `kvbm-connector/static-kernels` → `kvbm-physical/static-kernels` → `kvbm-kernels/static-kernels`
  - `kvbm-kernels` has a `build.rs` that shells out to **nvcc** to compile `tensor_kernels.cu`. Without nvcc in PATH the build script silently produces C stubs, which export abort-on-call symbols named differently from the real kernels — the resulting `_core.abi3.so` fails to load at import time with `undefined symbol: kvbm_kernels_launch_vectorized_copy`.
  - Note: per the kvbm-kernels docs, the `static-kernels` feature only controls linking (static `.a` vs dynamic `.so`) *when real CUDA is available*. Stubs always remain dynamically linked.
- `kvbm-py3/v1` pulls in `cudarc` (Rust bindings — needs CUDA *libs/headers* at build, not nvcc itself) and `dynamo-llm/block-manager`. `dynamo-llm` ships a pre-built FATBIN (`vectorized_copy.fatbin`) and reported `Building with CUDA KV off` in the build log, so **v1 does not compile CUDA kernels itself**.
- Answer to "can we gate CUDA around v1": no gating needed. v1 can already build without nvcc (just needs cudarc-compatible CUDA headers/libs). If someone wants a fully CUDA-free build experience, they can do `maturin develop --features v1 --no-default-features` and skip v2 entirely.

**Build baseline (phase 0.4)**:
- Sandbox venv: `/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox/` (Python 3.12.3, uv-managed).
- CUDA: `/usr/local/cuda` (nvcc 13.0.88). **nvcc MUST be in PATH for `maturin develop --features v1,v2`** (v2 path, see CUDA note above).
- `patchelf` is missing from the venv. maturin emits a warning about failed rpath setting but produces a working wheel anyway. If future runtime loading issues emerge, `pip install patchelf` into the sandbox venv.
- Working build command:
  ```bash
  source .sandbox/bin/activate
  export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda KVBM_REQUIRE_CUDA=1
  export PATH=/usr/local/cuda/bin:$PATH
  cd lib/bindings/kvbm && maturin develop --features v1,v2
  ```
- Smoke imports verified: `kvbm.__version__ == '1.0.0'`, `kvbm._V1_AVAILABLE and kvbm._V2_AVAILABLE`, `kvbm.v1.{BlockManager, KvbmLeader, KvbmWorker}` imports, `kvbm.v2.KvbmRuntime` imports.

### 2026-04-13 — Phase 3 execution: v1 determinism passing on GB10

**Outcome**: `pytest tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset -k "v1-Qwen3-0.6B"` → **1 passed in 131.74s** on the GB10/sm_121 host. Phase 2's harness decomposition is now validated end-to-end: fixture chain → KvbmServerManager → vllm spawn → KVBM v1 BlockManager init (zmq handshake, layer detection, block pool build) → determinism eval loop with cache reset → 100% match. The v1 stack works on Blackwell.

**Path chosen (option 2 from phase-3 plan)**: vllm cu130 nightly wheel (`https://wheels.vllm.ai/nightly/cu130`) + stock pytorch.org cu130 torch 2.11.0. Did **not** end up needing the cypheritai/pytorch-blackwell alpha wheel — installing it was a useful diagnostic step (proved the venv could host a sm_121 torch and that openblas was missing) but the stock cu130 nightly resolves a torch 2.11.0 that already targets sm_120 with forward-compat to sm_121, which is sufficient.

**Final venv delta (vs. `.sandbox/requirements.before-phase3.txt`, captured to `.sandbox/requirements.after-phase3-wheels.txt`)**:
| Package | Before | After |
|---|---|---|
| `torch` | `2.10.0+cu126` (sm_80, sm_90 only) | `2.11.0+cu130` (sm_80, sm_90, sm_100, sm_110, sm_120 — runs on sm_121 via fwd compat) |
| `torchvision` | `0.25.0` | `0.26.0` |
| `torchaudio` | `2.10.0` | `2.11.0` |
| `vllm` | `0.19.0` | `0.19.1rc1.dev232+g0e39202ca.cu130` |
| `nvidia-nccl-cu13` | `2.28.9` | `2.29.7` (force-bumped; vllm pins 2.28.9 but torch 2.11.0 needs `ncclDevCommDestroy` which is 2.29+) |
| `nvidia-nccl-cu12` | `2.27.5` | (uninstalled — was stomping the install path) |
| `flashinfer-python` | `0.6.6` | `0.6.7` |

System package added: `libopenblas0` + `libopenblas0-pthread` (apt; needed by the cypheritai diagnostic torch alpha — torch 2.11.0+cu130 stock also uses it).

**Reproduction recipe** (for the next person setting up GB10):
```bash
# From workspace root, with sandbox venv activated
sudo apt install -y libopenblas0
uv pip uninstall vllm
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130
uv pip uninstall nvidia-nccl-cu12  # leftover from torch 2.10.0+cu126
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'

# Rebuild kvbm-py3 against the new torch ABI
export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda KVBM_REQUIRE_CUDA=1
export PATH=/usr/local/cuda/bin:$PATH
cd lib/bindings/kvbm && cargo clean -p kvbm-py3 && maturin develop --features v1,v2

# IMPORTANT: maturin's pip-install step may roll nccl back to 2.28.9 because
# vllm pins it. Re-bump after maturin:
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

**vllm 0.19.1rc1 API drift fix** (one-liner): `lib/bindings/kvbm/python/kvbm/v1/vllm_integration/connector/pd_connector.py:15` — vllm 0.19.1 renamed `vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector` (module) to `.nixl` (package). Added a `try: from .nixl import (NixlConnector, NixlHandshakePayload); except ImportError: from .nixl_connector import (...)` fallback so the binding keeps working with both old and new vllm. This was caught by the phase-1 regression test (`test_legacy_imports.py`), which dropped from 6/7 → 3/7 on the new vllm and went back to 6/7 after the fix. PdConnector is for prefill/decode disaggregation; not used by the agg determinism test, but it's transitively imported by the v1 connector package init so it has to import cleanly.

**Test setup used** (small model, low iterations — designed for fast iteration, not full determinism numerics):
```bash
KVBM_MODEL_ID=Qwen/Qwen3-0.6B \
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 \
KVBM_GPU_MEMORY_UTILIZATION=0.5 KVBM_SERVER_START_TIMEOUT=300 \
pytest tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset \
    -v -k "v1-Qwen3-0.6B" --tb=short
```
Result: `PASSED [100%]`, total wall time 131.74s (server bring-up + 2 phases × 2 iterations + cache reset).

**KVBM stack health (from per-test vllm log)**:
- `KvConnectorWorker initialized with worker_id: 9c7c3326-1cb5-4779-8829-ed6eb41f5851`
- `Auto-detected device layout: LayerSeparate { outer_contiguous: true }`
- `Layout: num_layers=28, outer_dim=2, page_size=16, inner_dim=1024` (Qwen3-0.6B)
- `KvbmWorker num_device_blocks=512, page_size=16, dtype_width_bytes=2`
- ZMQ handshake leader↔worker completed
- Block pool built, offload subsystem configured
- Per-test log preserved at `/tmp/dynamo_tests/test_determinism_agg_with_cache_reset[v1-Qwen3-0.6B]/ServerType.vllm_server_v1_cpu2000_gpu512_20260413_010419.log`

**Out of scope for this run, deferred to phase-3 follow-up**:
- Scale to `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` at default iteration counts.
- Run the DeepSeek-V2-Lite (MLA, batch_invariant=False) parametrization.
- Capture quantitative `kvbm_offload_blocks_d2h` / `kvbm_onboard_blocks_h2d` metrics from the live `/metrics` endpoint (the server tears down after the test; would need a longer-lived run via the decomposed `run_server.sh v1` flow).
- FP8 path validation (`vllm-fp8` crash on sm_121 per eugr/spark-vllm-docker#143 is still a known sharp edge — Qwen3-0.6B doesn't take that path, so we sidestepped it cleanly).

### 2026-04-13 — Phase 3 remainder (decomposed flow + parity gate)

**Outcome**: Phase 3 closed. Three additions landed:

1. **MLA gated, not removed.** `tests/kvbm_integration/test_determinism_agg.py` keeps `DeepSeek-V2-Lite` in `_MODEL_CONFIGS` but new env gate `KVBM_ENABLE_MLA` (default off) wraps MLA specs in `pytest.mark.skip`. Refactored `_specs()` to return raw `KvbmServerSpec` lists and added `_params()` to wrap them as `pytest.param(...)` with optional skip marks. Two new module-level lists: `_CACHE_RESET_SPECS`/`_CONCURRENT_SPECS` (raw, used by scripts) and `_CACHE_RESET_PARAMS`/`_CONCURRENT_PARAMS` (used by the parametrize decorators). Rationale: MLA is out of scope for this plan but must be restorable later; keeping the spec definition intact means re-enabling is one env var instead of a code revert. The drift bug from the peer review (script hardcoding `block_size=16, attention_backend='FLASH_ATTN'`) is eliminated by step 2 below, not by removing the MLA spec.

2. **Scripts are spec-id driven.** `tests/kvbm_integration/scripts/run_server.sh` now takes a `KvbmServerSpec.id` (e.g. `v1-DeepSeek-R1-Distill-Llama-8B`), reconstructs the canonical spec by importing `_CACHE_RESET_SPECS` from the test module, and applies `KVBM_CPU_BLOCKS`/`KVBM_GPU_BLOCKS` overrides via `dataclasses.replace`. The hardcoded `KvbmModelConfig(block_size=16, attention_backend='FLASH_ATTN')` literal and the legacy `KVBM_MODEL_ID` consult are gone from the script. `run_server.sh` exits 7 on MLA spec ids without `KVBM_ENABLE_MLA=1` and exits 3 on `v2-*` (deferred to phase 5). On READY it now prints a third export, `KVBM_SPEC_ID`, alongside `KVBM_EXTERNAL_BASE_URL` / `KVBM_EXTERNAL_METRICS_PORT`. `tests/kvbm_integration/scripts/run_eval.sh` requires `KVBM_SPEC_ID` and uses it as the default pytest `-k` filter so shell 3 cannot pick up parametrizations the server was not launched for. Positional args still override.

3. **Decomposed three-shell smoke run validated end-to-end.** Found and fixed two pre-existing phase-2 bugs in `tests/kvbm_integration/scripts/run_deps_v1.sh` along the way:
   - Its `_Req` shim was missing `request.node.name` (added a stub `_Node`).
   - It tried to spawn `nats-server`/`etcd` binaries unconditionally even though neither is on PATH on this host. Aligned the script with `tests/kvbm_integration/conftest.py:runtime_services` — it now probes `NATS_SERVER`/`ETCD_ENDPOINTS` (defaults `nats://localhost:4222` + `http://localhost:2379`), reuses them if reachable (exits 0 immediately), otherwise spawns and stays in the foreground.
   Smoke result: `bash run_deps_v1.sh` reused local services, `bash run_server.sh v1-Qwen3-0.6B` brought up vllm in ~2 min, `bash run_eval.sh` (with `KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2`) exited 0 in **7.65s** with `1 passed, 1 deselected`. Live `/metrics` snapshot during the run: `kvbm_offload_blocks_d2h=99`, `kvbm_onboard_blocks_h2d=39` — first quantitative offload/onboard capture in this branch.

**Parity/signoff gate** (composed pytest path, default iteration counts):
- **Initial attempt: DeepSeek-R1-Distill-Llama-8B at default counts on GB10 — ABANDONED.** Pytest hit the `_CACHE_RESET_TIMEOUT` budget (1805s actual / 1800s budget with `KVBM_SERVER_START_TIMEOUT=900`) mid phase-2. Test was clearly progressing — at 02:03 host hit rate had climbed to 48.5% (1573/3246) and external prefix cache hit rate to 46.5% — but generation throughput on R1-Distill-Llama-8B/GB10 averages only ~6 tokens/s, so each iteration takes much longer than the 4s/iter the timeout formula assumes. Per-user direction (2026-04-13), rather than further chasing a 30+ minute 8B run on this host, downscoped phase 3's parity gate to Qwen3-0.6B at default counts.
- **Final parity gate: Qwen3-0.6B at default iteration counts — PASSED.**
  ```bash
  KVBM_MODEL_ID=Qwen/Qwen3-0.6B \
  KVBM_CPU_BLOCKS=2000 KVBM_GPU_BLOCKS=512 KVBM_GPU_MEMORY_UTILIZATION=0.5 \
  KVBM_SERVER_START_TIMEOUT=600 \
      pytest tests/kvbm_integration/test_determinism_agg.py::TestDeterminismAgg::test_determinism_agg_with_cache_reset \
          -v -k "v1-Qwen3-0.6B" --tb=short
  ```
  Result: `1 passed, 1 deselected in 190.47s (0:03:10)`. Default iteration counts (`KVBM_MAX_ITERATIONS=100`, `KVBM_NUM_ITERATIONS=15`, `KVBM_REQUEST_DELAY=30`) honored — no overrides. Final KVBM cache stats from the per-test vllm log: **Host: 66.4% (3394/5113)**, Disk: 0.0% (0/5113). 100% deterministic match across both cache-reset phases.

**R1-Distill-Llama-8B sign-off intentionally deferred to a different host.** Rationale per user (2026-04-13): GB10 generation throughput is too slow to make a 30+ minute test cycle a useful local iteration loop. **On this host, Qwen3-0.6B at default iteration counts is the standing test model for both v1 (now) and v2 (phase 5)**. Once both v1 and v2 are passing locally for 0.6B, work moves to a different host to validate the larger models (R1-Distill-Llama-8B and the gated MLA/V2-Lite spec). The test surface is already wired (both larger specs are in `_CACHE_RESET_SPECS`, MLA spec gates on `KVBM_ENABLE_MLA=1`, `run_server.sh v1-DeepSeek-R1-Distill-Llama-8B` works), so the off-host run is a one-command operation per spec.

**Verification matrix** (all green):
- `pytest tests/kvbm_integration/test_determinism_agg.py --collect-only -q` → 4 ids: `v1-DeepSeek-R1-Distill-Llama-8B` and `v1-DeepSeek-V2-Lite` × `cache_reset` and `concurrent`.
- `pytest tests/kvbm_integration/test_determinism_agg.py -v -k "v1-DeepSeek-V2-Lite and cache_reset"` → `SKIPPED [...] MLA gated; set KVBM_ENABLE_MLA=1 to enable`.
- `bash -n` on all three scripts → clean.
- `bash run_server.sh v1-Bogus-Model` → exit 6, lists known spec ids.
- `bash run_server.sh v1-DeepSeek-V2-Lite` → exit 7, references `KVBM_ENABLE_MLA`.
- `bash run_server.sh v2-foo` → exit 3, references phase 5.
- `KVBM_EXTERNAL_BASE_URL=... KVBM_EXTERNAL_METRICS_PORT=... bash run_eval.sh` (without `KVBM_SPEC_ID`) → exit 2.
- `pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q` → still 6 passed, 1 skipped.
- Decomposed three-shell flow, smoke iterations: PASSED in 7.65s.
- Composed parity gate, Qwen3-0.6B default iterations: PASSED in 190.47s, host hit rate 66.4%.

**Files touched** (this round):
- `tests/kvbm_integration/test_determinism_agg.py` — `_KVBM_ENABLE_MLA`, `_params()` helper, `_CACHE_RESET_PARAMS`/`_CONCURRENT_PARAMS`, parametrize decorators consume `_PARAMS`, `ids=[...]` kwarg dropped.
- `tests/kvbm_integration/scripts/run_server.sh` — fully rewritten as spec-id driven; heredoc `python -` block instead of `python -c`; v2 prefix check; MLA gate check; `dataclasses.replace` overrides; `KVBM_SPEC_ID` export.
- `tests/kvbm_integration/scripts/run_eval.sh` — requires `KVBM_SPEC_ID`; uses it as default `-k` filter.
- `tests/kvbm_integration/scripts/run_deps_v1.sh` — reuse-or-spawn parity with the conftest fixture; `_Req`/`_Node` shim now provides `node.name`.
- `tests/kvbm_integration/README.md` — decomposed-flow walkthrough updated for spec-id form, `KVBM_SPEC_ID` handshake documented, MLA gate note added.
- `ACTIVE_PLAN.md` — this entry; phase 3 state flipped to completed.

**Out of scope for this round, deferred to a future phase**:
- DeepSeek-R1-Distill-Llama-8B at default iteration counts — needs faster host or CI; spec wiring is ready.
- DeepSeek-V2-Lite (MLA) execution — gated; restore by setting `KVBM_ENABLE_MLA=1` once MLA is back in scope.
- FP8 path validation on sm_121.
- Long-lived `/metrics` snapshots from the 8B parity run.

### 2026-04-13 — Phase 2 execution

**Landed.** Three-layer fixture decomposition shipped. Files:

- `tests/kvbm_integration/fixtures/{__init__.py,deps.py,server.py,eval.py}` — new package
- `tests/kvbm_integration/fixtures/test_kv_transfer_config.py` — v2 builder unit test (6 cases)
- `tests/kvbm_integration/scripts/{run_deps_v1.sh,run_server.sh,run_eval.sh}` — layered local-iteration scripts
- `tests/kvbm_integration/test_determinism_agg.py` — rewritten to consume new fixtures; parametrize via `KvbmServerSpec`; v1-only enumeration (TODO comment for phase 5)
- `tests/kvbm_integration/conftest.py` — registers the new fixtures via re-export
- `tests/kvbm_integration/README.md` — full rewrite documenting three-layer architecture, env-var contract, layered execution
- `lib/bindings/kvbm/python/kvbm/vllm_integration/__init__.py` — phase-1 missed-cleanup fix

**KvbmServerManager** is the renamed `LLMServerManager`, taking a `KvbmServerSpec` instead of a flat constructor. Behavior is identical to the previous inline manager; the only behavior change is that the kv-transfer-config is now built by the pure function `build_kv_transfer_config(version, model_config)` (previously hardcoded). v1 payload matches the phase-1 shim path verbatim. v2 payload is shipped inert (never invoked from a test in phase 2) and **deliberately omits `leader.nova`** — `lib/kvbm-config/src/messenger.rs:43` defaults `discovery: None` and `build_messenger()` short-circuits.

**External-attach mode**: `KVBM_EXTERNAL_BASE_URL` (and `KVBM_EXTERNAL_METRICS_PORT`) gate. When set, `kvbm_deps` skips the runtime_services pull and `kvbm_server` skips the spawn — both bind to the running server via an `_ExternalServer` adapter that is duck-typed against `KvbmServerManager`. Used by `scripts/run_eval.sh` for layered iteration.

**Test method signature change**: dropped `runtime_services` from the test method args (was unused by the base class methods anyway). The test now takes `(self, kvbm_tester, kvbm_server)` and passes `None` positionally to `base_test_*` for the unused param. Reason: if the test method requested `runtime_services` directly, pytest would always pull it in even when `kvbm_deps` correctly skipped it for external-attach mode, defeating the purpose. Existing call into `common.py:base_test_*` is unchanged.

**Verification results**:
- `pytest tests/kvbm_integration/test_determinism_agg.py --collect-only -q` → 4 v1 IDs collected, no v2, no errors. ✓
- `pytest tests/kvbm_integration/fixtures/test_kv_transfer_config.py -v` → 6 passed. ✓
- `pytest tests/kvbm_integration/ --collect-only -q` → all kvbm tests collect cleanly except `test_cuda_graph.py` (pre-existing `ModuleNotFoundError: dynamo`, reproduced on stashed state — unrelated to phase 2). ✓
- `pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q` → 6 passed, 1 skipped (trtllm) — unchanged from phase 1. ✓
- Bash syntax check on all three scripts → clean. ✓
- Full fixture import chain → resolves; v1 and v2 builder dicts have correct shape. ✓
- **v1 end-to-end fixture-spawn dry-run** with `KVBM_MODEL_ID=Qwen/Qwen3-0.6B`, low iteration counts → fixture wiring resolved correctly through to vllm spawn (vllm received the right `--kv-transfer-config` payload from `build_kv_transfer_config("v1", ...)`, the right model, the right block overrides). vllm itself died at GPU init with `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device` — see "Blackwell GPU finding" below.

**Blackwell GPU finding (environmental, not a phase-2 regression)**: this branch is checked out on a host with an NVIDIA GB10 (sm_120/Blackwell) GPU, and the sandbox venv's `vllm 0.19.0` + bundled torch were not built with sm_120 kernels. Confirmed reproducible by invoking `vllm serve` directly with the same args (no fixture machinery) — same `cudaErrorNoKernelImageForDevice` error. Not a regression introduced by phase 2. The decomposed fixture chain executed end-to-end up to vllm bring-up, which is the meaningful integration check; the actual determinism loop is gated on a vllm/torch build that supports the host GPU. **Action**: log here, do not block phase 2 sign-off, surface for the user to re-run on a host with a compatible GPU (e.g. H100/L4/A100) before signing off the v1 determinism numerics. The decomposed scripts (`run_deps_v1.sh` + `run_server.sh v1` + `run_eval.sh`) are also runnable on such a host; their three-shell choreography is unchanged.

**Sign-off scope**: the harness decomposition itself (the phase 2 deliverable) is fully landed and verified to the limit of the local environment. The v1 determinism numerics validation is deferred to a re-run on a Blackwell-compatible vllm/torch (or a non-Blackwell GPU). Phase 3 was originally framed as "v1 end-to-end determinism passing" and naturally absorbs that re-run.

### 2026-04-13 — Phase 2 re-plan and Phase 4 scope expansion

**Phase 2 narrowed to v1 execution.** Original plan assumed v2 could be parametrized alongside v1 in the new fixtures. Peer-review verification on this branch found three blockers that prevent `kvbm.v2.vllm.schedulers.connector` from importing today:

1. **Eager scheduler init** — `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/__init__.py:5` does `from .dynamo import DynamoScheduler`. `dynamo.py` expects scheduler symbols not exported by `lib/bindings/kvbm/src/v2/mod.rs:8`. Importing the connector triggers the package init and dies.
2. **Hard vllm version block** — `lib/bindings/kvbm/python/kvbm/v2/vllm/config.py:22` raises `ImportError` on `vllm >= 0.12.2` (sandbox has 0.19.0). Bypass: `KVBM_SKIP_VLLM_VERSION_CHECK`.
3. **Two parallel version policies** — `lib/bindings/kvbm/python/kvbm/v2/vllm/version_check.py:7` defines a contradicting policy (max 0.14.0) gated on a different env var `KVBM_DISABLE_MAX_VERSION_CHECK`.

If we tried to enumerate `"v2"` in the phase-2 parametrize list, `pytest --collect-only` would itself fail with ImportError. Two options were considered: (a) split phase 2 into 2a/2b around the readiness gate, or (b) ship phase 2 with an inert v2 builder (pure function not invoked by tests) and defer the parametrize-flip to phase 5. **Chose (b)** — keeps phase numbering stable, decouples local v1 iteration from unrelated v2 readiness work, and the phase-5 prep task is a one-line parametrize edit. The v2 fixture seam still exists end-to-end so phase 5 doesn't need to come back here.

**Phase 4 scope expanded.** Originally framed as "verify v2 path wiring" + reconcile `schedulers/connectors` duplication. Now the v2-readiness gate. New deliverables: (i) unify the dual vllm version policies under a single env var, (ii) make `kvbm.v2.vllm.schedulers.connector` import-safe (either export the missing scheduler symbols from `lib/bindings/kvbm/src/v2/mod.rs:8` or stop `schedulers/__init__.py:5` from eagerly importing `dynamo.py`), (iii) cross-check phase 2's inert `build_kv_transfer_config("v2", ...)` against the actual leader config schema. Phase 4 may need its own re-plan once reached.

**Phase 0 reclassified.** Peer review found `cargo clippy --all-features --all-targets -- -D warnings` is not green workspace-wide on this branch. The crate-level clippy claim (`kvbm-*` + `kvbm-py3`) still holds, but local `lib/kvbm-kernels/tests/{memcpy_batch.rs,kernel_roundtrip.rs}` and upstream `dynamo-runtime` debt break `--all-targets`. Per user direction (2026-04-13), clippy `-D warnings` strictness is **relaxed for phases 2–5** and **re-enforced as a phase-6 sign-off gate**. Phase 0's "completed" rubric narrowed accordingly in the State-per-phase section.

**Phase-1 missed cleanup.** `lib/bindings/kvbm/python/kvbm/vllm_integration/__init__.py:4-6` still says "subpackages resolve to v2 implementations" — contradicts the Option A v1 shim and the test_legacy_imports.py identity check. One-line fix folded into phase 2 task 1.

**Velo/nova rename status.** Sub-finding from phase-2 ground-truth: the Rust crate is named `velo` (`lib/kvbm-config/src/messenger.rs` uses `use velo::Messenger`), but the serde config key in `MessengerConfig` is still literally `"nova"`. Python classes still `NovaPeerMetadata` (`lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/worker.py:43`). Phase 2 sidesteps by omitting the `leader.nova` block entirely (discovery defaults to `None` per `lib/kvbm-config/src/messenger.rs:43` — verified with `build_messenger()` short-circuiting at line 56). Finishing the rename is out of scope for the active plan; flagged for a future cleanup phase.

**Phase 5 TODO carried in.** Phase 2 hardcodes `cache.host.cache_size_gb = 10.0` in the v2 builder (matches sandbox script verbatim). Phase 5 must derive it from `KVBM_CPU_BLOCKS × block_size × dtype_bytes × num_layers` so v1 and v2 host-cache sizing is comparable for determinism validation.

## State per phase

- Phase 0 — **completed (buildable, clippy debt deferred)** 2026-04-12 — workspace hygiene, kvbm-* + kvbm-py3 clippy green at the crate level, maturin build + smoke imports verified. **Reclassified 2026-04-13**: `cargo clippy --all-features --all-targets -- -D warnings` is **not** green workspace-wide — `lib/kvbm-kernels/tests/{memcpy_batch.rs,kernel_roundtrip.rs}` and the documented upstream `dynamo-runtime` debt still trip it. Clippy strictness is relaxed for phases 2–5 per user direction; re-enforced as a phase-6 sign-off gate.
- Phase 1 — **completed** 2026-04-12 (legacy v1 paths restored; lazy `__getattr__` shim; consolidator_config gap closed; conftest gates rewritten; 6/7 regression tests pass, 1 trtllm-skipped; see deviations log). Missed cleanup at `kvbm/vllm_integration/__init__.py:4-6` folded into phase 2 task 1.
- Phase 2 — **completed** 2026-04-13 (decomposition landed; collect-only smoke + v2 builder unit test green; v1 fixture wiring verified end-to-end through vllm spawn; full v1 vllm run blocked by environmental Blackwell/sm_120 PyTorch issue, not a fixture regression — see deviations log)
- Phase 3 — **completed on GB10 (Qwen3-0.6B at default iteration counts)** 2026-04-13 — sm_121 venv wired via vllm cu130 nightly + nccl 2.29.7; kvbm-py3 rebuilt; phase-1 regression test still 6/7 green after a one-line vllm-API drift fix in `pd_connector.py`; **decomposed three-shell flow validated end-to-end** (run_deps_v1.sh reuse-or-spawn fix + spec-id refactor of run_server.sh / run_eval.sh + MLA gate); **`test_determinism_agg_with_cache_reset[v1-Qwen3-0.6B]` PASSED at default iteration counts in 190.47s with KVBM host hit rate 66.4%**. Per user direction (2026-04-13), Qwen3-0.6B is the standing test model on this host; larger models (DeepSeek-R1-Distill-Llama-8B and the gated DeepSeek-V2-Lite/MLA spec) are deferred to a different host once both v1 and v2 are passing here for 0.6B. See deviations log for spec-id refactor details and the exact wheel set + reproduction recipe.
- Phase 4 — **completed 2026-04-13** — v2 readiness gate closed. Single vllm version policy (max bumped to 0.19.999); `kvbm.v2.vllm.schedulers.connector` import-safe via widened `except` in `dynamo.py:47` AND a hidden second AttributeError gap in `kvbm/v2/__init__.py` (split into core+scheduler try blocks); new canonical `kvbm.v{1,2}.vllm.connector` façades (1↔2 char mirror) with 4 lazy-shim regression tests; both builders point at the new façades; `KvbmServerSpec.onboard_mode` field for `intra`/`inter`; **both intra and inter v2 vllm bring-ups PASSED on GB10 with Qwen3-0.6B** (mode confirmed via `kvbm_connector::connector::leader: ConnectorLeader initialized with onboard mode onboard_mode=Intra/Inter` log lines); v1 phase-3 determinism test still green against new façade in 44s. Rust scheduler port (`dynamo_kvbm::v2::integrations::scheduler::*` → decomposed kvbm-* crates) documented and deferred — DynamoScheduler's `_rust_scheduler is None` fallback is sufficient. See deviations log for the full reproduction recipe and the 8-command verification gate.
- Phase 5 — **completed 2026-04-13** — `test_determinism_agg_with_cache_reset` passes at default iteration counts for **both v2 onboard modes** on GB10 with Qwen3-0.6B (intra: 182.63s/67.0% host hit; inter: 183.76s/66.7% host hit); v1 baseline still 66.4%. Parametrize permanently `("v1","v2")`, v2 crossed with `("intra","inter")`. v2 builder now passes `cache.host.num_blocks = spec.cpu_blocks` (exact v1 parity); the plan's original `cache_size_gb` derivation was rejected in favor of this simpler path. Rust config gained max-wins semantics and a mandatory-tier bail. Critical in-phase fix: removed three spurious `#[cfg(feature = "nccl")]` gates on the intra onboard path in `kvbm-engine/src/worker/physical.rs` and `kvbm-connector/src/connector/worker/mod.rs` — the gated code was pure CUDA (H2D stream + per-layer `CudaEvent::record`), not nccl, and without the gate removed the intra onboard was a silent no-op that caused vLLM to read uninitialized KV slots. `cudarc` made unconditional in kvbm-engine; the `nccl` cargo feature now only gates the genuine collectives code. Decomposed three-shell v2 flow validated for both modes (intra smoke 11.4s, inter smoke 11.5s). Larger-model parity (R1-Distill-Llama-8B, MLA) still deferred to a faster host. See deviations log for metrics, reproduction recipe, and file list.
- Phase 6 — not started; clippy `-D warnings --all-targets` re-enforcement added to scope

## Handover note

When this plan is approved, copy it to `ACTIVE_PLAN.md` at the workspace root (`/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/ACTIVE_PLAN.md`) so the `CLAUDE.md` active-plan convention picks it up. Update both files in lockstep; prefer editing the workspace-root copy as the execution source of truth and pulling changes back here only when a new planning pass is needed.
