# Planner Plugin Architecture

This directory implements **DEP-XXXX Dynamo Planner Plugin Architecture** (v11).
All new plugin-related code lives here; existing `core/` is read-only
until PR 11 cleanup.

**👉 First, read `DEP-XXXX_HANDOFF.md` at the repo root for state and
workflow.** This file is the module-local convention summary.

## Module layout

```
plugins/
├── proto/v1/                  # PR 1: gRPC schema + generated stubs
│   ├── plugin.proto           # SOURCE OF TRUTH
│   ├── plugin_pb2*.py / .pyi  # generated, check-in
│   └── README.md              # schema invariants + evolution policy
├── types.py                   # PR 1: Pydantic v2 mirror of proto messages
├── _proto_bridge.py           # PR 1: bidirectional Pydantic↔proto converter
├── lifecycle.py               # PR 6: PluginLifecycle ABC (Bootstrap/Reset)
├── scheduler.py               # PR 3: PluginScheduler (cache invalidation table)
├── transport/                 # PR 2: 3 transports (in_process / uds / grpc)
│   ├── base.py                # PluginTransport ABC
│   ├── in_process.py / uds.py / grpc_remote.py
│   ├── _grpc_base.py          # uds + grpc shared mixin (call/close/error mapping)
│   ├── _method_dispatch.py    # method name → grpc stub map
│   ├── _mtls.py               # MtlsConfig (cert-manager 3-key convention)
│   ├── config.py              # TransportConfig + factory
│   ├── errors.py              # PluginCallError hierarchy
│   └── README.md              # decision tree + Threat Model + sync plugin red line
├── clock.py                   # PR 2: Clock ABC + WallClock + VirtualClock
├── merge/                     # PR 4: type-aware merge + chain-augment
│   ├── type_aware.py          # PROPOSE/RECONCILE/CONSTRAIN merge (sync)
│   ├── chain_augment.py       # PREDICT stage (async; layered partial-merge)
│   └── types.py               # PluginResult / ChainAugmentOutcome / MergeOutcome
├── registry/                  # PR 3: PluginRegistry + CircuitBreaker + heartbeat
│   ├── server.py              # PluginRegistryServer (4 RPC + register_internal)
│   ├── circuit_breaker.py     # per-plugin CLOSED/OPEN/HALF_OPEN state machine
│   ├── heartbeat_monitor.py   # liveness monitor (skips in_process)
│   ├── config.py              # PluginRegistrationConfig + build_auth_validator
│   ├── errors.py              # AuthError + registry error hierarchy
│   └── auth/                  # 5 auth validators (PR 3 v1 + PR 3.5)
│       ├── static_secret.py   # PR 3 v1 (shared-secret map)
│       ├── multi.py           # PR 3 v1 (MultiSourceAuth fan-out)
│       ├── base.py            # PR 3 v1 (AuthValidator ABC + AllowUnauthenticatedAuth)
│       ├── k8s_sa_token.py    # PR 3.5: K8s TokenReview
│       └── spiffe_jwt.py      # PR 3.5: SPIRE JWT-SVID via JWKS
├── builtins/                  # PR 6: 5 real in-process builtin plugins
│   ├── base.py                # BuiltinPluginBase (shared regression store access)
│   ├── load_predictor.py      # PREDICT stage (priority 1)
│   ├── throughput_propose.py  # PROPOSE stage (priority 10; sets AT_LEAST)
│   ├── load_propose.py        # PROPOSE stage (priority 5; reads throughput lower bound)
│   ├── reconcile.py           # RECONCILE stage (v1 passthrough)
│   └── budget_constrain.py    # CONSTRAIN stage (emits AT_LEAST/AT_MOST, NOT SET)
└── orchestrator/              # PR 5 + PR 7: orchestrator + engine adapter
    ├── orchestrator.py        # LocalPlannerOrchestrator (register_internal + install_regressions + bootstrap_plugins)
    ├── pipeline.py            # 4-stage pipeline driver (M-1 zip + M-4 skip + M-7 no-wait_for)
    ├── in_process_loader.py   # load InProcessPluginSpec entries
    ├── internal_register.py   # module-level register_internal convenience
    └── engine_adapter.py      # PR 7: OrchestratorEngineAdapter (EngineProtocol impl)
```

## Hard rules

### Forbidden patterns

| Pattern | Why | Use instead |
|---|---|---|
| `time.time()` / `time.monotonic()` / `asyncio.sleep` directly | breaks replay / VirtualClock | injected `Clock` from `clock.py` |
| `asyncio.wait_for(asyncio.gather(...))` wrapping a stage | redundant; per-plugin timeout in transport already handles it (M-7 OR option) | just `await asyncio.gather(...)` and trust per-plugin `request_timeout_seconds` |
| bare `except:` catching transport errors | masks plugin contract violations | catch `PluginCallError` subclass from `transport/errors.py` |
| editing generated `*_pb2.py` files | hand edits will be overwritten | edit `plugin.proto`, run `tools/build/gen_planner_proto.sh` |
| editing `core/state_machine.py` / `core/load_scaling.py` / `core/throughput_scaling.py` | breaks G3 fixture parity guard (PR 5/6/7/8 全程 read-only PSM) | wait for PR 11 cleanup |
| calling `state_machine.on_tick()` directly from new code | breaks dual-path abstraction; PSM tick is sync, orchestrator tick is async | go through `_ensure_engine().tick(...)` (returns `EngineProtocol`, async everywhere) |

### Required conventions

| Convention | Where | How |
|---|---|---|
| Pydantic v2 (NOT v1) | all new schemas | `from pydantic import BaseModel, ConfigDict, Field` |
| Pydantic strict mode | new config classes | `model_config = ConfigDict(extra="forbid")` |
| Proto field optionals | proto3 fields needing unset semantics | `optional float foo = 1;` (e.g. `PredictionData.predicted_*` per v11 G-1) |
| Test markers (4 required) | every new test file in `tests/plugins/` | `pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.planner]` |
| Async plugin methods | when doing IO | `async def Predict(self, req): ...` (sync OK only for pure CPU; see sync red line) |
| Lock-step proto + Pydantic + bridge | adding any new message | edit proto + types.py + _PYD_TO_PROTO + round-trip test in same PR |

### Sync plugin red line (`InProcessTransport`)

Sync (`def`) plugin methods are dispatched via `asyncio.to_thread`,
which uses the default ~32-thread pool. **Sync plugins MUST NOT do
blocking IO** (HTTP, file, `time.sleep > 100ms`). A few misbehaving
sync plugins exhaust the pool and stall the orchestrator.

If your plugin needs IO, write `async def`.

## Build

```bash
# Regenerate proto stubs after editing plugin.proto
tools/build/gen_planner_proto.sh

# CI mode: regenerate to temp + diff vs committed (no drift check)
tools/build/gen_planner_proto.sh --check
```

## Test

All test files using markers `[pre_merge, planner, gpu_0, unit]` are
auto-picked by the existing `planner-test` CI job (see
`.github/workflows/pr.yaml`). **No `.yml` edits needed** for new tests.

```bash
# Run all DEP-XXXX tests
cd components/src && python -m pytest dynamo/planner/tests/plugins -q

# Run full planner-test marker scope (matches CI)
python -m pytest dynamo/planner/tests/ -m "pre_merge and planner and gpu_0" -q \
  --ignore=dynamo/planner/tests/unit/test_prometheus.py \
  --ignore=dynamo/planner/tests/unit/test_diagnostics_recorder.py
```

## Schema invariants (CANNOT change without breaking algorithms)

- **`PredictionData.predicted_*` fields are `optional float`** — PR 4
  chain-augment partial-merge uses `HasField()` to distinguish
  "I assert 0" from "no opinion". Removing `optional` silently breaks
  the layered-predictor pattern.
- **CONSTRAIN `SET` is silently dropped at runtime** (NOT register-time
  rejected). Orchestrator emits `plugin_constrain_set_dropped` audit +
  Prometheus counter.
- **REJECT > final priority** in same stage. If any plugin returns
  `RejectResult`, the stage short-circuits even when `final=true`
  plugins are present.
- **`final=true` semantics differ between PREDICT and PROPOSE/RECONCILE**:
  - PROPOSE/RECONCILE: priority number smallest wins
  - PREDICT (chain-augment): first `final=true` in chain wins (chain
    is priority-descending → first = lowest priority); enforces
    "final plugin must have lowest priority number" via runtime
    detection in PR 4
- **Empty `oneof result` = plugin contract violation**, NOT silent ACCEPT

## PR status snapshot (2026-04-23)

| PR | Subject | Status |
|---|---|---|
| 1 | proto + Pydantic types + bridge | shipped |
| 2 | Transport (in_process / uds / grpc) + Clock | shipped |
| 3 | PluginRegistry + CircuitBreaker + Scheduler (v1: `static_secret` + `allow_unauthenticated` only) | shipped |
| 3.5 | follow-up: `K8sSATokenAuth` + `SpiffeJwtAuth` | shipped 2026-04-23 |
| 4 | type-aware merge + chain-augment | shipped |
| 5 | `LocalPlannerOrchestrator` + 4-stage pipeline | shipped |
| 6 | 5 real builtin plugins + 10 G3 scenarios (target was 36; 26 remain as optional expansion) | shipped (6-7 PsmShim deleted 2026-04-23) |
| 7 | `NativePlannerBase` dual-path + `use_orchestrator` flag (default false) | 10/11 (7-9 e2e blocked on runtime infra) |
| 8 | observability + replay | not started |
| 10/11 | flip flag default + delete PSM | not started |

## Pointers

| Topic | Doc |
|---|---|
| Full design | `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md` v11 (repo root) |
| Implementation plan + cross-check | `DEP-XXXX_Implementation_Breakdown_zh.md` v2 |
| Per-PR sub-task tables | `DEP-XXXX_PR{1..8}_Detailed_zh.md` |
| Resume work | `DEP-XXXX_HANDOFF.md` |
| Orchestrator rollout runbook | `docs/components/planner/orchestrator-rollout.md` |
| Schema details | `proto/v1/README.md` |
| Transport decision tree + Threat Model | `transport/README.md` |
