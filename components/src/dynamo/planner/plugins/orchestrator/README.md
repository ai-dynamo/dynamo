# LocalPlannerOrchestrator (DEP-XXXX PR 5 — partial)

Composes the PR 1-4 pieces into a single per-tick pipeline driver:

```
┌─ LocalPlannerOrchestrator ──────────────────────────────────────┐
│                                                                 │
│  registry        scheduler        circuit_breaker        clock  │
│      │               │                   │                │     │
│      └─── run_pipeline (4 stages) ───────┘                │     │
│                                                           │     │
└───────────── tick(ctx, baseline) → PipelineOutcome ───────┘─────┘
```

## Pipeline stages

| # | Stage | Merge | What flows forward |
|---|---|---|---|
| 1 | PREDICT | `chain_augment` (priority-desc) | merged `PredictionData` → `ctx.predictions` |
| 2 | PROPOSE | `type_aware_merge`, `set_allowed=True` | proposal → `ctx.proposal` + baseline for RECONCILE |
| 3 | RECONCILE | `type_aware_merge`, `set_allowed=True` | proposal → `ctx.proposal` + baseline for CONSTRAIN |
| 4 | CONSTRAIN | `type_aware_merge`, `set_allowed=False` (SET dropped) | proposal → EXECUTE decision |
| 5 | EXECUTE | — (decision, not RPC) | `PipelineOutcome.execute_action` |

The baseline flows **through** the stages (PR 4 cross-task §4):

- PROPOSE uses the caller's initial baseline (current worker counts).
- RECONCILE uses `_proposal_to_baseline(PROPOSE.proposal)`.
- CONSTRAIN uses `_proposal_to_baseline(RECONCILE.proposal)`.

EXECUTE returns one of:

| `execute_action` | Meaning |
|---|---|
| `apply` | Caller should enact `final_proposal.targets` via PlannerConnector |
| `skip_no_targets` | **M-4** — CONSTRAIN produced zero targets; audit `execute_skipped_no_targets` |
| `skip_short_circuit` | Some stage produced a `RejectResult`; no EXECUTE |
| `skip_tick_timeout` | Whole-tick deadline exceeded; audit `tick_timeout_total` |

## Strong constraints (v11)

- **M-1** — stage call results are paired with plugins via
  `zip(plugins, results)`. Never reach through
  `result.plugin_id`/`result.priority`; raw stage responses have no
  plugin back-reference.
- **M-4** — when CONSTRAIN's `proposal.targets` is empty (every PROPOSE
  plugin ACCEPTed + empty baseline), return `skip_no_targets` instead
  of `apply`ing an empty proposal. `PipelineOutcome.audit_events`
  carries `"execute_skipped_no_targets"`.
- **M-7** — **no stage-level `asyncio.wait_for`** wrapping
  `asyncio.gather`. Per-plugin timeouts live inside
  `PluginTransport.call` (`request_timeout_seconds`). The single
  `asyncio.wait_for` in `pipeline.py` wraps the whole-tick body.
  `tests/plugins/orchestrator/test_pipeline.py::test_M7_...` is a
  grep+AST regression test.

## Regression model access (v11 § Q2)

Current PR 5 shape: a simple `dict[str, Any]` owned by the orchestrator,
exposed via `get_regression(kind)` / `update_regression(kind, model)`.
All access happens on the event-loop main task; no locks.

`snapshot_regression` is intentionally not present — YAGNI per v1.3.
When an async builtin needs to hold a regression-model reference across
an `await`, reintroduce `snapshot_regression` as `copy.deepcopy`.

## What's IN this partial PR 5

| Sub-task | Status |
|---|---|
| 5-1 directory skeleton | ✅ |
| 5-2 `LocalPlannerOrchestrator` + regression accessors | ✅ (PipelineContext-native `tick`) |
| 5-3 scheduler integration | ✅ (self-wires via PR 3's bilateral subscription) |
| 5-4 4-stage pipeline + M-1 / M-4 / M-7 | ✅ |
| 5-5 `register_internal` delegate | ✅ |
| 5-6 `load_in_process_plugins` importlib loader | ✅ |
| 5-9 concurrency / failure / timeout tests | ✅ |
| 5-10 this README | ✅ |

## What's deferred to a supervised session

| Sub-task | Reason |
|---|---|
| 5-7 placeholder builtins wrapping PSM mixins | Requires reading + wrapping `core/state_machine.py` / `load_scaling.py` / `throughput_scaling.py` internals. Needs someone who can exercise real workloads to catch drift. |
| 5-8 G3 fixture parity replay | Requires 5-7 + the fixture lock golden jsonl files. The main doc tags this as the most important acceptance test — getting it wrong silently is worse than landing it incrementally. |
| `TickInput` → `PipelineContext` bridging + `PlannerEffects` projection | Per the DEP, this adapter belongs in **PR 7** (NativePlannerBase dual-path). The orchestrator's public `tick` API in this PR is PipelineContext-native so the skeleton can be exercised without touching existing adapter code. |

## Testing layout

```
tests/plugins/orchestrator/
├── conftest.py                      — ctx_factory, StubPlugin
├── test_orchestrator_lifecycle.py   — 12 tests: construct / register / tick / shutdown / regression
├── test_pipeline.py                 — 11 tests: stages, M-1/M-4/M-7, HOLD_LAST, misuse audit
├── test_concurrency.py              — 5 tests: gather parallelism, timeouts, circuit integration
├── test_in_process_loader.py        — 5 tests: importlib + kwargs + error paths
└── _fake_in_process_plugin.py       — test-only plugin class
```

Run with:

```bash
cd components/src
python -m pytest dynamo/planner/tests/plugins/orchestrator -q
# → 33 passed
```

## Cross-references

| Topic | File |
|---|---|
| 4-stage pipeline source | `pipeline.py` |
| Orchestrator + regression accessors | `orchestrator.py` |
| Module-level `register_internal` | `internal_register.py` |
| In-process plugin config loader | `in_process_loader.py` |
| `PluginRegistryServer` | `../registry/server.py` |
| `PluginScheduler` + cache table | `../scheduler.py` |
| Merge algorithms | `../merge/README.md` |
