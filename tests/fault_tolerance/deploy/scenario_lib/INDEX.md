# Scenarios

YAML-driven test scenarios consumed by `test_router_modes.py` and (future)
sibling test files. Each subdirectory == one test "kind"; the scenario
YAML's `kind:` field must match the parent directory name.

## Layout

```
scenario_lib/
├── _schema.py           # dataclass schema for Scenario, Deployment, Router, etc.
├── _loader.py           # strict YAML → Scenario loader
├── _runtime.py          # bridge: auto-discovers Event/Report/Check subclasses,
│                        # builds the scenario's event sequence + reports
├── INDEX.md             # this file
├── router_memory/       # consumed by test_router_memory
├── endurance/           # consumed by test_endurance (long-running mocker)
└── admission_control/   # (future) consumed by test_admission_control
```

## How it works (in one paragraph)

A scenario YAML carries `deployment`, `router`, `admission`, `load`,
`events`, `reports`, `checks`, and `expectations`. The runner loads the
YAML, builds a DeploymentSpec from the templated DGD, applies router
and admission knobs as env-vars on the right services, then either uses
the explicit `events:` block or generates the default
`WaitForModelReady → PodMemoryPoller → for-each-rung(StartLoad +
WaitForLoadCompletion)` sequence. Event/Report/Check classes are
auto-discovered by walking `Event.__subclasses__()` /
`Report.__subclasses__()` / `Check.__subclasses__()` at import time —
no hardcoded registry. Adding a new class anywhere in `events.py` /
`reports.py` / `checks.py` makes it usable in YAML immediately as
`kind: NewClassName`.

## Adding a scenario

1. Pick the kind (= which test function should run it). Put the YAML
   under `scenario_lib/<kind>/<descriptive-filename>.yaml`.
2. Name convention: `<router_mode>__<traffic_shape>__<topology>__<backend>.yaml`.
   E.g. `kv_prefix_aware__partial_prefix__disagg__vllm.yaml`.
3. Required top-level fields: `kind`, `name`, `deployment`, `router`,
   either `load` (with `rungs:`) or explicit `events:`.
4. Optional: `admission`, `reports`, `checks`, `expectations`.
5. `pytest --collect-only` should now show the test ID. If not, the
   loader raised a validation error — read the test output for the
   line/field that didn't parse.

## Adding a new kind (new test type)

1. Create `scenario_lib/<new_kind>/` and add at least one YAML there.
2. In `test_router_modes.py` (or a sibling test file), add:
   ```python
   _MY_SCENARIOS = discover_scenarios("<new_kind>")
   @pytest.mark.parametrize("scenario_path", _MY_SCENARIOS, ids=[p.name for p in _MY_SCENARIOS])
   async def test_my_kind(runtime_env, request, scenario_path):
       await _run_yaml_scenario(scenario_path)
   ```

## Current scenarios

### router_memory/

Memory-leak characterization across router configurations and workload
shapes. Anchored by the cycle-6 finding that
`DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT > 0` flips a binary leak ~27 MB/min.

| Scenario | Router | Shape | Expected slope (MB/min) |
|---|---|---|---|
| `kv_load_aware__partial_prefix__disagg__vllm.yaml` | KV-aware, no prefix scoring | partial_prefix (37% hit) | 1.5–4.5 (negative reference) |
| `kv_prefix_aware__partial_prefix__disagg__vllm.yaml` | KV-aware + OVERLAP=1.0 | partial_prefix | 22–32 (positive — production-default leak) |
| `least_loaded__partial_prefix__disagg__vllm.yaml` | LeastLoaded | partial_prefix | 1.5–3.5 (control: leak NOT in LL path) |
| `kv_prefix_aware__no_prefix__disagg__vllm.yaml` | KV-aware + OVERLAP=1.0 | no_prefix (random) | 25–80 (S-N1: H1 lever) |
| `kv_prefix_aware__same_prefix__disagg__vllm.yaml` | KV-aware + OVERLAP=1.0 | same_prefix (shared 2K-token system prompt) | 2–28 (S-N2: H1 lever) |

### endurance/

Long-running mocker-backed scenarios for surfacing slow drift (memory
leak, queue backlog) that only shows up after hours. Uses the new
`PeriodicSnapshot` event to drop timestamped artifact bundles into
`<log_dir>/snapshots/t{minutes:04d}m_<iso>/` so analysis can proceed
in-flight.

| Scenario | Duration | Snapshot interval | Notes |
|---|---|---|---|
| `endurance_mocker_smoke.yaml` | ~30 min | 5 min | Validates the machinery; coffee-break smoke test |
| `endurance_mocker_long.yaml`  | ~6 h    | 30 min | Production-default router knobs; checks leak slope vs router_memory expectation |

To run a paired A/B (e.g. OVERLAP=1.0 vs =0.0), launch dynamo-ft twice
in two namespaces with the two scenario filenames. No automatic
pairing framework — manual two-namespace launch keeps the framework
simple and resource accounting clear.

Set `DYN_ENDURANCE_MAX_HOURS=N` to raise the snapshot ceiling (default 8h).

## Hypothesis matrix (recap from observe's leak-rootcause handoff)

| Hypothesis | What's leaking | Predicted scaling |
|---|---|---|
| **H1** — radix tree node retention | Block nodes in DashMap + Arc<RwLock<Block>> allocations | prefix uniqueness × ISL |
| **H2** — SchedulingRequest retention | Per-request HashMap allocations | request count |
| **H3** — per-worker tracking growth | Per-(worker × request) tracking entries | request × N workers |

The pairing of `no_prefix` vs `same_prefix` vs `partial_prefix` on the
SAME router config (`kv_prefix_aware`) discriminates H1. Future
scenarios in `long_isl/` and `high_qps/` shapes will discriminate H2
and worker-count lever for H3.
