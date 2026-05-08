# Planner Orchestrator Rollout Runbook

This runbook covers the operational aspects of migrating the planner
from its legacy `PlannerStateMachine` (PSM) path to the new plugin
orchestrator path introduced by DEP-XXXX. Both paths coexist in the
same binary; a feature flag selects which one the planner uses at
runtime.

This doc is written for **operators running a Dynamo planner** — SRE /
platform teams doing the rollout, not the developers who wrote the
code. If you're troubleshooting a specific incident or code change,
see `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md` in the repo
root for design context.

---

## TL;DR

- **Default behaviour is unchanged.** The feature flag
  `planner.scheduling.use_orchestrator` defaults to `false`; upgrading
  binaries doesn't switch paths. Nothing happens until you set the
  flag.
- **To enable the new path**: set
  ``planner.scheduling.use_orchestrator: true`` in your planner config
  + restart (or process reload). The planner will construct a
  `LocalPlannerOrchestrator` with 5 builtin plugins instead of the
  PSM. A ready-to-use sample DGD lives at
  `components/src/dynamo/planner/tests/manual/perf_test_configs/disagg_8b_planner_orchestrator.yaml`
  — copy that file (or apply it directly with `kubectl apply -f`) and
  diff against `disagg_8b_planner.yaml` to see the one-field opt-in.
  Setting the flag via DGDR is also supported under
  `spec.features.planner.scheduling.use_orchestrator` (see [planner
  guide](planner-guide.md#schedulingengine-selection)).
- **To roll back**: set the flag to `false` + reload. **Takes under a
  minute**, no redeploy, no data migration.
- **Known regressions on the orchestrator path** (see
  [Known gaps](#known-gaps) below):
  - Numeric diagnostic fields (`estimated_ttft_ms`,
    `predicted_num_req`, etc.) are **not yet emitted** — your
    Prometheus dashboards show 0 for them. **Scaling decisions
    themselves are byte-identical to PSM** (see `scale_to` parity
    tests); only observability is degraded.
  - Decision-reason log lines show `n/a` instead of PSM's
    per-component reason text.

---

## What is `use_orchestrator`?

`PlannerStateMachine` (PSM) is the original monolithic class that
decides how many prefill / decode replicas to run each tick. DEP-XXXX
decomposes it into a 5-plugin pipeline (`BuiltinLoadPredictor`,
`BuiltinThroughputPropose`, `BuiltinLoadPropose`, `BuiltinReconcile`,
`BuiltinBudgetConstrain`) driven by a `LocalPlannerOrchestrator`.

The two paths are wired behind an `EngineProtocol` abstraction:

```
                 ┌─ use_orchestrator = false (default)
                 │      → _PSMEngineAdapter(PlannerStateMachine)
 NativePlannerBase.run()
                 │      → OrchestratorEngineAdapter(
                 └─ use_orchestrator = true           5 builtin plugins )
```

Both paths consume the same `TickInput` (traffic metrics, FPM samples,
worker counts) and produce the same `PlannerEffects.scale_to`
(recommended replica counts). The ONLY differences are:

1. **Who computes the decision**: PSM's monolithic `on_tick` vs. the
   5-plugin pipeline.
2. **Diagnostics field values**: PSM populates `TickDiagnostics`
   (numeric fields + reason strings); orchestrator currently emits an
   empty `TickDiagnostics` (Prometheus-metric migration pending).

Scaling decisions match across both paths for all G3 regression
scenarios (`test_dual_path_parity.py`). `next_tick.at_s` also
matches — main-loop cadence is preserved.

---

## Staged rollout

### Stage 0 — Dev sanity

Run the planner locally (or in a dev cluster) with
`scheduling.use_orchestrator: true`. Easy mode
(`optimization_target: throughput` or `latency`) is the lowest-risk
starting point — it doesn't use regression models.

Success criteria:

- Planner starts without error.
- `run()` loop ticks at the expected cadence (check
  `load_adjustment_interval` / `throughput_adjustment_interval`
  config values).
- `connector.add_component` / `remove_component` calls happen when
  workload crosses the easy-mode static thresholds.
- No exceptions in the log.

Expected degradations (normal, not a bug):

- Prometheus `dynamo_planner_estimated_ttft_ms` / `_itl_ms` /
  `predicted_*` gauges show `0` on this path.
- Log line `[summary] ... load_reason=n/a throughput_reason=n/a
  est_ttft=0.0ms est_itl=0.0ms`.

If dev passes: proceed to staging. If it fails: flip flag back off,
file an issue with the planner log + config + the failing scenario.

### Stage 1 — Staging

Enable on a staging cluster that runs representative workload.
SLA-mode (`optimization_target: sla`) is valid here because PR 7
sub-task 7-4 wired regression bootstrap through the adapter.

Duration: **at least one full traffic cycle** (business-hours peak +
off-hours trough). For most deployments this is 24-48 hours.

Observe:

- **Scaling decisions track PSM**: if you can compare logs from a
  parallel PSM run (e.g. staging-A on PSM, staging-B on orchestrator
  with same traffic), the `scale_to` values per tick should be
  identical. Any drift is a bug — stop and investigate before going
  further.
- **Tick cadence is stable**: `tick_duration_seconds` histogram P99
  stays under `scheduling.tick_max_duration_seconds` (default 30s) ×
  0.5. If P99 grows over time, something is leaking (likely a plugin
  cache or a Prometheus label cardinality issue).
- **No tick timeouts**: `tick_timeout_total` counter stays at 0. A
  non-zero value means the pipeline exceeded the whole-tick deadline
  — investigate before proceeding.

### Stage 2 — 1 production cluster (canary)

Pick your smallest-blast-radius production planner. Enable for 1
cluster. Monitor for **at least one full release cycle** (typically
1-2 weeks).

Rollback trigger criteria — any of:

- Scaling decisions diverge from PSM in parallel comparison.
- P99 `tick_duration_seconds` exceeds 0.5× `tick_max_duration_seconds`
  sustained over 1 hour.
- `tick_timeout_total > 0` — the pipeline missed a deadline.
- Unexpected exceptions in the planner log that aren't present on
  PSM-path clusters.
- Prometheus alert `dynamo_planner_plugin_circuit_open_total > 0`
  (once PR 8 observability ships).

### Stage 3 — Full fleet

Only after Stage 2 runs clean for the observation window. Roll out by
cluster class (dev → staging → prod) or by region, never all at once.

Keep the PSM path available in the binary for 2-3 Dynamo releases after
full rollout; only then is PSM code removed (see `DEP-XXXX_Implementation_Breakdown_zh.md`
for PR 10/11 cleanup plan).

---

## Rollback

**If something goes wrong at any stage:**

1. Set `planner.scheduling.use_orchestrator: false` in the planner
   config.
2. Reload the planner process (K8s: restart the planner Pod;
   systemd: `systemctl restart dynamo-planner`).
3. The planner will reconstruct its engine under the PSM path on
   startup. Expected downtime: **under 1 minute**.

No data migration is needed. Regression model observations accumulated
under the orchestrator path are discarded — PSM rebuilds from its
scenario-specific `load_benchmark_fpms` bootstrap. This is safe because
regression-model state is not persisted across planner restarts on
either path.

**If rollback itself fails**: the PSM path is the pre-DEP-XXXX code,
untouched by any orchestrator-path commit. Any breakage of the PSM
path is an unrelated regression — follow your standard planner
incident runbook.

---

## Known gaps

### 1. Empty diagnostics in orchestrator path

**Resolved by DEP-XXXX PR 8** (2026-04-26). The orchestrator path
now populates `TickDiagnostics.load_decision_reason*` and
`estimated_*_ms` from `BuiltinLoadPropose._last_load_diagnostics` (PR 8
sub-task 8-9) and emits the full plugin-era metric surface (sub-tasks
8-2 / 8-3 / 8-5):

- `dynamo_planner_load_scaling_decision` enum tracks
  `insufficient_data` / `no_change` / `scale_up` / `scale_down`
  identically to PSM.
- `dynamo_planner_estimated_{ttft,itl}_ms` populated from the
  regression model when it has sufficient data.
- 19 new `dynamo_planner_*` series cover plugin lifecycle (family 2),
  RECONCILE/CONSTRAIN clamps (family 3), EXECUTE outcomes (family 5),
  and tick scheduling (family 6). Catalog in [observability.md](observability.md).
- Structured `AUDIT` log events on `dynamo.planner.audit` logger;
  catalog in [audit-events.md](audit-events.md).

K8s-validated end-to-end: planner produced a real `scale_up`, called
`set_component_replicas`, operator created a new GPU-backed Pod.

**Caveat**: pre-deployment profiling data must still be present
(`profile_results_dir` populated, or `pre_deployment_sweeping_mode !=
"none"`) for the regression to fit and `est_*_ms` to be non-zero.
Without profiling data the path correctly reports
`load_reason=insufficient_data`, matching PSM's behaviour.

### 2. No full-stack automated test coverage

`tests/integration/test_dual_path_parity.py` validates engine-level
output parity (scale_to, next_tick) for 10 regression scenarios, but
does not drive a full `NativePlannerBase.run()` loop end-to-end with
real connector / runtime wiring. DEP-XXXX PR 7 sub-task 7-9 adds that
coverage; until then, Stage 1 staging observation is your primary
confidence signal.

### 3. `GlobalPlannerConnector.set_predicted_load` wire (orchestrator path)

Resolved in PR 7 sub-task 7-7. The orchestrator path now calls
`connector.set_predicted_load(num_requests, isl, osl)` once per tick
between `engine.tick` and `_apply_effects`, guarded by:
1. `scheduling.use_orchestrator=True`
2. `connector` exposes a callable `set_predicted_load` (today only
   `GlobalPlannerConnector` — `KubernetesConnector` / `VirtualConnector`
   don't, so the call silently skips)
3. At least one `TickDiagnostics.predicted_*` field is populated —
   the orchestrator adapter fills these from
   `ChainAugmentOutcome.prediction` whenever the PREDICT stage
   produced output.

Caveat: the legacy PSM path never called `set_predicted_load`
(pre-existing gap, verified via `grep -r "set_predicted_load"
components/` — no production caller). PR 7 deliberately does **not**
retrofit the PSM path; fixing that is out of scope. Behavior on
`use_orchestrator=False` is byte-exact unchanged.

`environment: global-planner` is now supported on the orchestrator
path — the flag is safe to enable once Stage 1 staging observation
(and your own workload's predictor output) validates the wire fires
as expected.

---

## Capacity planning

`scheduling.tick_max_duration_seconds` (default 30s) is the outermost
deadline wrapping the entire 4-stage pipeline. The default suits most
deployments, but validate against your workload before production.

### Monitoring baseline

| Metric | Healthy range | Alert threshold |
|---|---|---|
| `dynamo_planner_tick_duration_seconds` P99 | `< 0.5 × tick_max_duration_seconds` | `> 0.8 × tick_max_duration_seconds` |
| `dynamo_planner_tick_timeout_total` | `0` | `> 1 / hour` |
| `dynamo_planner_plugin_latency_seconds` P99 | `< request_timeout_seconds × 0.5` | `> request_timeout_seconds × 0.8` |

(The plugin latency metric ships with PR 8 observability; the tick
metrics exist today.)

### Tuning decision tree

If P99 `tick_duration_seconds` approaches 50% of `tick_max_duration_seconds`:

- **One plugin RPC is slow**: raise that plugin's
  `request_timeout_seconds` OR optimize the plugin implementation OR
  move it to in-process transport.
- **Multiple plugins approaching their per-plugin timeout**: raise
  `tick_max_duration_seconds` (e.g. 30s → 60s); verify this doesn't
  cascade to missed scaling windows.
- **One stage consistently slow**: check the stage's plugin list —
  too many plugins fanning out in `asyncio.gather`?

If `tick_timeout_total > 0`: the pipeline missed a whole-tick
deadline. Likely a systemic deadlock (plugin blocked on a
never-returning RPC that didn't respect `request_timeout_seconds`).
**Stop, roll back, file an incident** — the orchestrator path is
designed to never let a single plugin exhaust the whole tick, so a
timeout indicates a deeper bug.

---

## Troubleshooting

### Symptom: planner crashes on startup with `use_orchestrator=true`

- Check the log for the exception. Most common: a mode subclass's
  `_bootstrap_regression` raised because `fetch_pre_deployment_metrics`
  couldn't reach the benchmark data source. On PSM path the same
  error is tolerated when `enable_throughput_scaling=False`; the
  orchestrator path matches this behaviour (PR 7 sub-task 7-4), so
  confirm the toggle config.
- Check the Pod's memory limit: the orchestrator path constructs 5
  plugin instances + a registry server + a scheduler. Memory overhead
  over PSM is small (< 10 MB) but non-zero.

### Symptom: scaling decisions differ between PSM and orchestrator clusters

- Confirm both clusters have the same `PlannerConfig`, same
  `PlannerCapabilities` (from MDC), and same model benchmark data.
- Check the planner log around the divergent tick for `[summary]`
  lines: `action` / `current` / `recommended` should be identical
  (even if reason text differs — orchestrator path says `n/a`).
- If `recommended` values differ: file a bug. `test_dual_path_parity.py`
  should have caught the drift; if it didn't, the test set is
  incomplete and the scenario needs adding.

### Symptom: log summary lines all say `n/a` / `0.0ms`

Expected. See [Known gaps §1](#1-empty-diagnostics-in-orchestrator-path).
Scaling decisions are unaffected.

### Symptom: `dynamo_planner_plugin_circuit_open_total` is non-zero

A builtin plugin has been tripped into the OPEN circuit state by
repeated failures. Check the log for the specific `plugin_id`. Common
causes:

- Plugin exception from a code bug (check the full stack trace).
- Plugin hit `request_timeout_seconds` repeatedly (consider raising).
- Plugin's regression model crashed (check `optimization_target=sla`
  scenarios + benchmark data quality).

Circuit breakers auto-recover (HALF_OPEN after `cooldown_seconds`,
back to CLOSED on one success), so this is usually transient — but
the underlying cause should be investigated.

---

## Validation before enabling

Before flipping `use_orchestrator=true` on anything you care about,
verify the CI tests these commit hashes depend on are green in your
build:

```bash
cd components/src
python -m pytest dynamo/planner/tests/plugins -q
# expect: all tests passing (currently 394 as of DEP-XXXX PR 7 7-8 ship)

python -m pytest dynamo/planner/tests/integration/test_dual_path_parity.py -v
# expect: 30 passed (scale_to + next_tick + initial_tick × 10 scenarios)

python -m pytest dynamo/planner/tests/ -m "pre_merge and planner and gpu_0" -q \
  --ignore=dynamo/planner/tests/unit/test_prometheus.py \
  --ignore=dynamo/planner/tests/unit/test_diagnostics_recorder.py
# expect: ~671 passed
```

If these don't pass in your build, do not enable the flag. The
dual-path parity matrix is the regression signal for "is this safe to
switch"; a failure there means the two paths diverge in ways that will
surface as production incidents after cutover.

---

## References

- DEP-XXXX main doc (Chinese): repo-root
  `DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md`
- PR 7 implementation breakdown: `DEP-XXXX_PR7_Detailed_zh.md`
- Dual-path parity test: `components/src/dynamo/planner/tests/integration/test_dual_path_parity.py`
- Orchestrator adapter: `components/src/dynamo/planner/plugins/orchestrator/engine_adapter.py`
- Feature flag definition: `components/src/dynamo/planner/config/planner_config.py` (`SchedulingConfig`)
