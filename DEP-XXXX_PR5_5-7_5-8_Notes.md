# PR 5 5-7 / 5-8 — Implementation Notes for Next Session

> Handoff from 2026-04-22 PR 5 partial session. Skeleton + pipeline
> driver + concurrency tests are **done**; this doc records everything
> needed to execute the deferred 5-7 (placeholder builtins) and 5-8
> (G3 fixture parity) without re-reading the whole repo.

## Green on arrival

```bash
pytest dynamo/planner/tests/plugins -q           # 305 passed
pytest dynamo/planner/tests/ -m "pre_merge and planner and gpu_0" -q \
  --ignore=dynamo/planner/tests/unit/test_prometheus.py \
  --ignore=dynamo/planner/tests/unit/test_diagnostics_recorder.py
# → 503 passed, 1 skipped, 11 deselected
```

`tests/plugins/g3_fixtures/test_g3_fixture_parity.py` is **already
green** — it runs PSM against the locked golden jsonl. Your job is to
add a sibling that runs the **orchestrator** through the same
scenarios and asserts the same byte-level equality.

---

## PSM public contract (read once, don't re-explore)

`PlannerStateMachine(LoadScalingMixin, ThroughputScalingMixin)` lives
at `components/src/dynamo/planner/core/state_machine.py:51`. Key API:

```python
class PlannerStateMachine:
    def __init__(self, config: PlannerConfig, caps: WorkerCapabilities): ...
    def initial_tick(self, start_s: float) -> ScheduledTick: ...
    def load_benchmark_fpms(self, ...) -> None: ...       # optional bootstrap
    def warm_load_predictors(self, observations: list[TrafficObservation]) -> None: ...
    def on_tick(self, tick: ScheduledTick, tick_input: TickInput) -> PlannerEffects: ...
```

### `on_tick` body — verbatim (state_machine.py:170-211)

```
effects = PlannerEffects()
_reset_diag()
if tick_input.worker_counts is not None: _update_inventory(tick_input.worker_counts)

# 1. Throughput runs first — sets _throughput_lower_bound_[p|d]
#    that load scaling later reads.
throughput_decision = None
if tick.run_throughput_scaling:
    if tick_input.traffic is not None:
        _observe_traffic(tick_input.traffic)
        throughput_decision = _advance_throughput(tick_input.traffic)
    _next_throughput_s = tick_input.now_s + config.throughput_adjustment_interval

# 2. Load runs second.
if tick.run_load_scaling:
    if tick_input.fpm_observations is not None:
        if not _is_easy:
            _observe_fpm(tick_input.fpm_observations)
        load_decision = _advance_load(tick_input.fpm_observations)
        if load_decision is not None:
            effects.scale_to = load_decision
    _next_load_s = tick_input.now_s + config.load_adjustment_interval

# 3. Load takes precedence; fall back to throughput.
if effects.scale_to is None and throughput_decision is not None:
    effects.scale_to = throughput_decision

effects.diagnostics = _build_diagnostics()
effects.next_tick = _next_scheduled_tick()
return effects
```

### The 5 mixin entry points the PR 5 5-7 placeholders wrap

| Mixin method | File:line | Returns | State touched |
|---|---|---|---|
| `_advance_throughput(traffic)` | throughput_scaling.py:35 | `Optional[ScalingDecision]` | predictors (`_num_req_predictor` / `_isl_predictor` / `_osl_predictor`); `_throughput_lower_bound_p/d` (SET when both toggles on); `_diag_throughput_*` |
| `_advance_load(obs)` | load_scaling.py:47 | `Optional[ScalingDecision]` | `_num_p_workers` / `_num_d_workers`; `_throughput_lower_bound_p/d` (READ); `_diag_load_*`; `_is_easy` |
| `_observe_traffic(traffic)` | state_machine.py:314 | None | feeds `_num_req_predictor` etc. |
| `_observe_fpm(obs)` | state_machine.py:297 | None | feeds `_prefill_regression` / `_decode_regression` / `_agg_regression` |
| `_update_inventory(counts)` | state_machine.py:274 | None | `_num_p_workers` / `_num_d_workers` |

### Critical inter-tick state

**Do NOT construct a fresh PSM per tick** — every entry point reads
state seeded by prior ticks:

1. **Predictors** (`_num_req_predictor`, `_isl_predictor`, `_osl_predictor`)
   — accumulate traffic observations; reset = wrong predictions.
2. **Regression models** (`_prefill_regression`, `_decode_regression`,
   `_agg_regression`) — accumulate FPM observations.
3. **Throughput lower bounds** (`_throughput_lower_bound_p/d`) —
   written by throughput stage, read by load stage later in the **same
   tick** AND surviving across ticks.
4. **Worker counts** (`_num_p_workers`, `_num_d_workers`) — updated
   from `_update_inventory`; read by load scaling as "current" count.
5. **`_next_throughput_s`, `_next_load_s`** — drive `_next_scheduled_tick`.

---

## Design tension the next agent must resolve

**PR 5 5-7 wording says**: each of the 5 placeholders constructs its
own `PlannerStateMachine(config, caps)` at plugin init. **This is
almost certainly wrong** — 5 independent PSM instances would each run
its own predictor / regression state and never see shared throughput
lower bounds. G3 parity will fail.

**Three options**:

### Option A — Single shared PSM instance (recommended)

Have the orchestrator (or a small bridge class) own ONE `PSMBridge`
holding a `PlannerStateMachine`. Each of the 5 placeholder plugins
keeps a reference and calls into the shared instance. Pipeline order
maps to PSM's internal order:

| Plugin | Delegates to |
|---|---|
| `load_predictor_placeholder` (PREDICT) | `PSMBridge.predict()` → calls `_predict_load()` + returns PredictionData |
| `throughput_propose_placeholder` (PROPOSE) | `PSMBridge.advance_throughput(traffic)` → wraps `_advance_throughput` return into `OverrideResult` |
| `load_propose_placeholder` (PROPOSE) | `PSMBridge.advance_load(fpm)` → wraps `_advance_load` return |
| `reconcile_placeholder` (RECONCILE) | Thin pass-through; type_aware_merge already gives the "load > throughput" precedence via priority |
| `budget_constrain_placeholder` (CONSTRAIN) | `PSMBridge.apply_budget(desired)` → calls `_apply_single_budget` / `_apply_global_budget` |

This preserves PSM's internal state machine while exposing it through
5 plugin seams.

### Option B — Single monolithic shim plugin

One plugin in PROPOSE that calls `PSM.on_tick(tick, tick_input)`
directly and emits the full `ScalingDecision` as an `OverrideResult`.
Simpler but doesn't exercise the pipeline's merge/chain-augment paths
(they'd all be no-ops). Parity-safe but less valuable as a skeleton.

### Option C — Strict PR 5 5-7 spec (5 × PSM)

5 independent PSMs. Will almost certainly fail G3 parity. Not recommended.

**My recommendation**: Option A, then add a `修订历史 v2.1` entry
explaining the deviation from the PR 5 doc's "each plugin constructs
its own PSM" language.

---

## 5-8 replay test — shape

Model on the existing `test_g3_fixture_parity.py`:

```python
# tests/plugins/orchestrator/test_g3_orchestrator_parity.py
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
async def test_g3_orchestrator_parity(scenario):
    # 1. Build orchestrator + register the 5 placeholders against a
    #    shared PSMBridge(scenario.make_config(), scenario.caps_factory())
    # 2. For each tick_input in scenario.ticks:
    #      - build PipelineContext from tick_input
    #      - outcome = await orchestrator.tick(ctx, baseline_from_worker_counts)
    #      - project outcome → PlannerEffects (scale_to, next_tick, diagnostics)
    #      - record the tuple
    # 3. Read golden fixture, compare via _compare_records.
    ...
```

Note: you'll need the **TickInput → PipelineContext + PlannerEffects
projection** bridging here (which I explicitly deferred from PR 5).
That's fine — doing it in the parity test validates it end-to-end in
one shot, which is what PR 7 will reuse.

### TickInput → PipelineContext bridge

```python
def tick_input_to_context(ti: TickInput) -> PipelineContext:
    return PipelineContext(
        request_id=f"tick-{ti.now_s}",
        decision_id=f"d-{ti.now_s}",
        observations=ObservationData(
            traffic=(
                TrafficMetrics(
                    duration_s=ti.traffic.duration_s,
                    num_req=ti.traffic.num_req,
                    isl=ti.traffic.isl,
                    osl=ti.traffic.osl,
                )
                if ti.traffic else None
            ),
            workers=(
                WorkerState(
                    ready_prefill=ti.worker_counts.ready_num_prefill,
                    ready_decode=ti.worker_counts.ready_num_decode,
                    expected_prefill=ti.worker_counts.expected_num_prefill,
                    expected_decode=ti.worker_counts.expected_num_decode,
                )
                if ti.worker_counts else None
            ),
            fpm=_fpm_observations_to_fpm_data(ti.fpm_observations),  # helper
        ),
    )
```

`FpmObservations` has `prefill: dict[(worker_id, dp_rank), FPM]`.
`FpmData` (PR 1) has `prefill_engines: dict[str, bytes]`. Key format
in the golden fixture is `"w1:0"` (see sample jsonl). So:

```python
def _fpm_observations_to_fpm_data(obs):
    if obs is None: return None
    def _pack(engines):
        if engines is None: return {}
        return {f"{wid}:{rank}": _fpm_to_bytes(fpm) for (wid, rank), fpm in engines.items()}
    return FpmData(prefill_engines=_pack(obs.prefill), decode_engines=_pack(obs.decode))
```

`_fpm_to_bytes` can pickle / msgspec / JSON-encode — as long as the
placeholder's `_observe_fpm` delegate knows how to decode it. If the
PSMBridge just caches the raw `FpmObservations` object alongside the
context, you can bypass encoding entirely for this session.

### Final proposal → PlannerEffects.scale_to

```python
def outcome_to_effects(outcome, psm_bridge) -> PlannerEffects:
    # diagnostics + next_tick come from psm_bridge (PSM's own state)
    diag = psm_bridge.build_diagnostics()
    next_tick = psm_bridge.next_scheduled_tick()

    if outcome.execute_action != "apply":
        return PlannerEffects(scale_to=None, next_tick=next_tick, diagnostics=diag)

    targets = outcome.final_proposal.targets
    num_prefill = next((t.replicas for t in targets if t.sub_component_type == "prefill"), None)
    num_decode = next((t.replicas for t in targets if t.sub_component_type == "decode"), None)
    return PlannerEffects(
        scale_to=ScalingDecision(num_prefill=num_prefill, num_decode=num_decode),
        next_tick=next_tick,
        diagnostics=diag,
    )
```

Compare against the golden via `_compare_records(expected, actual, scenario_name)` from `dump_tool.py:183`.

---

## Scenarios (6 total, already covered by fixtures)

| Name | Mode | Toggles | Target |
|---|---|---|---|
| `baseline_disagg_throughput_only_sla` | disagg | T only | sla |
| `disagg_load_throughput_sla` | disagg | L + T | sla |
| `disagg_load_only_latency_easy` | disagg | L only | latency (easy) |
| `agg_throughput_only_sla` | agg | T only | sla |
| `prefill_throughput_only_sla` | prefill | T only | sla |
| `decode_throughput_only_sla` | decode | T only | sla |

Start with `agg_throughput_only_sla` — it's simplest (single engine
type, no load-path complexity).

---

## Expected first-shot failures + fixes

Based on the PSM code review:

1. **`diagnostics` fields missing** because placeholders don't populate
   PSM's `_diag_*` scratch fields → fix: have placeholders call through
   the shared PSM; when PSM returns, its `_diag_*` fields are set.
2. **`next_tick` drift** if placeholders compute cadence independently
   of `_next_throughput_s` / `_next_load_s` → fix: pull from PSM.
3. **`scale_to` disagreement** when load + throughput both produce
   decisions — PSM uses "load wins, else throughput"; the orchestrator
   pipeline's priority ordering must match (make load plugin's
   priority smaller than throughput plugin's).
4. **FPM encoding round-trip** — if you pickle → unpickle, object
   identity differs. Easier to keep FPM objects as Python refs in a
   side-channel dict and not round-trip through `FpmData.prefill_engines`
   bytes for the placeholder path.

---

## Starter prompt for next session

> Continue DEP-XXXX PR 5 5-7 / 5-8. Read `DEP-XXXX_PR5_5-7_5-8_Notes.md`
> for the PSM contract + recommended design (shared PSMBridge, not 5
> independent PSMs). Implement in this order:
>   1. `orchestrator/psm_bridge.py` — single shared PSM holder with
>      `advance_throughput` / `advance_load` / `build_diagnostics` /
>      `next_scheduled_tick` accessors.
>   2. 5 placeholder plugins in `plugins/builtins/` delegating to the
>      bridge.
>   3. `tests/plugins/orchestrator/test_g3_orchestrator_parity.py`
>      starting with ONE scenario (`agg_throughput_only_sla`). Get it
>      green before expanding.
>   4. Add remaining 5 scenarios.
>   5. Record the Option-A deviation from PR 5 5-7's "each plugin
>      constructs its own PSM" wording in the PR 5 doc 修订历史.
> Keep 305-plugin-test baseline green throughout.
