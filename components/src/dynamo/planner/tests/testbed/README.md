# Power Planner Stress Testbed

Pure in-memory stress testbed for the Dynamo Power Planner. No GPUs, no
Kubernetes, no Prometheus server required — runs on any developer laptop in
< 30 s (α-class) or < 2 min (γ-class).

See the design document for full rationale:
`docs/design-docs/powerplanner-testbed-design.md`

---

## Quick start

```bash
# Install test dependencies (from repo root)
pip install -e "components/[testbed]"

# Run all α-class scenarios (fast, ~30 s)
pytest components/src/dynamo/planner/tests/testbed/ -v -m "testbed and not gamma"

# Run the full suite including γ-class (requires dynamo.llm / mocker)
pytest components/src/dynamo/planner/tests/testbed/ -v -m testbed

# Run a single scenario by name
pytest components/src/dynamo/planner/tests/testbed/test_scenarios.py::test_alpha[A1_power_under_estimate_decode] -v

# CLI: run one scenario, dump CSV + plot
python -m dynamo.planner.tests.testbed.runner \
    --scenario A1 \
    --csv out/A1.csv \
    --plot out/A1.png \
    --prom-textfile /tmp/A1.prom

# CLI: run the whole catalog
python -m dynamo.planner.tests.testbed.runner --all --csv-dir out/
```

---

## Architecture

```
testbed/
├── __init__.py                 # import guard (blocks pynvml)
├── conftest.py                 # blocks real K8s writes (session-scoped)
├── clock.py                    # deterministic virtual clock
├── scenarios.py                # Pydantic models + YAML loader (extends: support)
├── runner.py                   # ScenarioRunner: dispatches α / γ
├── recorder.py                 # TickSnapshot + TickHistory (CSV / Prom / plot)
├── assertions.py               # assertion DSL evaluator
│
├── synthetic_fleet.py          # α: truth model (power + latency + capacity)
├── fake_prometheus.py          # SHARED: FakePrometheusClient
├── fake_actuator.py            # α: FakeActuator (PlannerConnector impl)
├── fake_aic.py                 # SHARED: FakeAIC + per-system Pareto tables
├── fake_planner_metrics.py     # SHARED: in-memory Counter/Gauge mocks
│
├── replay/                     # γ-class extension (mocker-based)
│   ├── synthetic_power_overlay.py
│   ├── replay_fake_actuator.py
│   └── power_aware_replay_adapter.py
│
├── traces/                     # γ trace fixtures
│   ├── placeholder_h200_disagg_1rps.jsonl
│   └── README.md
│
├── systems/                    # per-SKU hardware constants
│   ├── h200_sxm.yaml
│   ├── h100_sxm.yaml
│   └── h100_pcie.yaml
│
├── scenarios/                  # 27-scenario catalog
│   ├── _base/                  # inheritance templates
│   └── A1_*.yaml … G3_*.yaml
│
├── test_scenarios.py           # pytest parametrize over scenarios/*.yaml
└── tests/                      # tests OF the testbed itself
    ├── test_fakes.py
    ├── test_overlay.py
    ├── test_scenarios_loadable.py
    └── test_self_consistency.py
```

### Test classes

| Class | Driver | Observability | Scheduler | Speed |
|-------|--------|---------------|-----------|-------|
| **α** | `SyntheticFleet` (in-memory truth model) | `FakePrometheusClient` | Deterministic (no Kubernetes) | < 1 s / scenario |
| **γ** | `dynamo-mocker` replay + `SyntheticPowerOverlay` | Same FakePrometheusClient (power from overlay) | Real mocker scheduler | 2–4 s / scenario |
| **real-AIC** | Real `aiconfigurator` perf database from a mounted sandbox | `MagicMock` metrics; in-process EMA loop | None (drives `update_correction()` directly) | < 20 s / suite |

#### real-AIC class

Opt-in only. Exercises `AICPowerOptimizer.optimize()` and the EMA drift loop
against a real AIC perf database with measured `power_w` data — the closest
thing to a production loop we can run without a real GPU. See §8 row 14 of
`powerplanner-design.md` for the defensive clamp this class regression-tests.

```bash
# Mount or symlink your AIC power-data tree at .aic_sandbox/systems/ first
# (h200_sxm.yaml + data/h200_sxm/...; same layout as aiconfigurator/systems/).
AIC_SANDBOX_DIR="$PWD/.aic_sandbox/systems" \
    pytest components/src/dynamo/planner/tests/testbed/tests/test_aic_real_data.py -v
```

What it verifies (parametrized across every SKU present in the sandbox):

* `AIConfiguratorPerfEstimator.estimate_perf` returns non-zero `power_w`
  for both prefill and decode (i.e. the sandbox actually has power data,
  not a TDP-fallback fixture).
* `AICPowerOptimizer.optimize()` produces a `PowerAwareConfig` whose
  per-GPU caps never exceed `TDP × _COEFF_MAX = 2 × TDP`.
* `aic_power_w_clamped_total{side=...}` fires iff AIC's raw `power_w`
  exceeded `TDP × 1.1` for that side. On the H200 vLLM 0.19.1 data this
  is exercised; on B200 TRT-LLM 1.3.0rc6 the data is dense enough that
  the clamp stays cold.
* The EMA loop converges in three regimes — well-calibrated → 1.0,
  over-predict → pegs at 0.5, under-predict → 1.5 — against H200's
  *real* AIC denominators.
* `should_reoptimize()` respects the hysteresis count under sustained
  SLA breach and stays silent under healthy load.

CI runs this class by setting `AIC_SANDBOX_DIR` in the job environment;
local invocations without the env var are silently skipped at module
import time with a one-line reason. The marker is `@pytest.mark.real_aic`.

γ-class skips automatically in two situations:

1. **No `dynamo._core` native binding** (e.g., fresh dev box without `maturin`):
   the runtime stub installs a no-op `dynamo._core` and `conftest.py`'s
   `pytest_collection_modifyitems` hook skips every `@pytest.mark.gamma` test.
   See Appendix C.10 in `docs/design-docs/powerplanner-testbed-design.md`.
2. **Older bridge API only** (`create_disagg` without `from_synthetic_disagg`):
   the α–γ cross-validation test (`test_alpha_gamma_agree_on_decode_drift`)
   skips because the placeholder trace fallback can't drive AIC drift. The
   three γ scenarios + `test_gamma_no_bias` still run. See Appendix D.7 in
   `docs/design-docs/powerplanner-testbed-design.md`.

Expected outcomes by environment:

| Environment | Pass | Skip |
|-------------|-----:|-----:|
| Local box (no Rust mocker) | 82 | 5 (all γ) |
| Dev pod (older `create_disagg` bridge) | 86 | 1 (α–γ cross-validation only) |
| Future CI box (newer `from_synthetic_disagg`) | 87 | 0 |

---

## Scenario catalog

| Group | Scenarios | What's stress-tested |
|-------|-----------|---------------------|
| A | A1–A6 | AIC drift / mis-calibration |
| B | B7–B11 | Actuation failures (NVML clamp, K8s RBAC, DaemonSet absent, frontend POST) |
| C | C12–C16 | Node / pod failures and recovery |
| D | D17–D21 | Observability / data-plane failures (Prometheus outage, stale, DCGM loss, MDC, window-cross) |
| E | E21–E25 | Budget / config edge cases |
| F | F26 | Drift threshold boundary |
| G | G1–G3 | γ-class: realistic scheduler-driven scenarios |

Total: 27 scenarios (24 α + 3 γ).

---

## Adding a new α scenario (< 1 hour)

1. Copy a base template:
   ```bash
   cp scenarios/_base/h200_disagg.yaml scenarios/X27_my_scenario.yaml
   ```

2. Edit the YAML — set `name:`, `description:`, `events:`, and `assertions:`.

3. Check the assertion field names are valid:
   ```bash
   pytest tests/testbed/tests/test_scenarios_loadable.py -v
   ```

4. Run your new scenario:
   ```bash
   pytest "test_scenarios.py::test_alpha[X27_my_scenario]" -v
   ```

### Available event types

Each event is a dict in `events:` with a `type:` discriminator. Single source
of truth: `scenarios.py` (the Pydantic `Event` union).

| `type` | Required fields | Effect |
|--------|----------------|--------|
| `bias_step` | `at_tick`, `signal`, `value` | One-shot bias change on `signal` (`power_bias_decode`, `power_bias_prefill`, `ttft_bias`, `itl_bias`, `capacity_bias`). Optional `auto_inject_window_cross: true` schedules a one-tick Prom-aggregation-window-cross at `at_tick+1`. |
| `bias_ramp` | `start_tick`, `end_tick`, `signal`, `from`, `to` | Linear ramp on `signal` over `[start_tick, end_tick]`. |
| `bias_sine` | `signal`, `amplitude`, `period_ticks`, `offset` (optional) | Sustained sinusoidal perturbation. |
| `actuation_fault` | `at_tick`, `duration_ticks`, `mode` | `mode ∈ {rbac_denied, nvml_low, nvml_high, daemonset_absent}`. |
| `node_down` | `at_tick`, `n_prefill_lost`, `n_decode_lost` | Yank replicas without going through the planner. |
| `node_up` | `at_tick`, `n_prefill_restored`, `n_decode_restored` | Restore lost replicas. |
| `prom_outage` | `at_tick`, `duration_ticks`, `signals:` (list) | Force `get_*` to return `None` for the listed Prometheus signals. |
| `prom_stale` | `at_tick`, `duration_ticks`, `lag_ticks` | Read returns the value from `tick − lag_ticks`. |
| `prom_window_cross_event` | `at_tick`, `signal`, `weight_old` | Models a Prom aggregation window that spans a cap or replica change. |
| `budget_change` | `at_tick`, `new_total_w` | Shrink / expand `total_gpu_power_limit` live. |
| `frontend_post_fault` | `at_tick`, `duration_ticks`, `failing_fraction` | Probabilistically fail `/busy_threshold` POSTs. |
| `mdc_unavailable` | `at_tick`, `duration_ticks` | MDC returns no `max_batched_tokens` (admission goes implied-only). |
| `aic_failure` | `at_tick`, `mode`, `n_consecutive` | `mode ∈ {empty_pareto, raises}`; resets after `n_consecutive` ticks. |

### Available assertion predicates

Every assertion needs **exactly one** of `at_tick:` / `always: true` /
`eventually_by_tick:`. The Pydantic validator rejects ambiguous or empty
predicates at load time (no more silent skips).

```yaml
# Point-in-time
- at_tick: 50
  field: c_power_d
  op: ">="
  value: 1.2

# Always (every tick)
- always: true     # MUST be `true`; bare `always:` is rejected
  field: projected_w
  op: "<="
  ref: "planner.total_gpu_power_limit"

# Eventually
- eventually_by_tick: 80
  field: sweep_fired
  op: "=="
  value: 1.0

# Expression (restricted AST eval; allowed roots: history, planner, counters, abs, min, max, len)
- expr: "history[-1].c_power_d > history[10].c_power_d"
  always: true
  description: "coefficient increases over time"
```

Valid `op` values: `==`, `!=`, `<`, `<=`, `>`, `>=`, `within`.
For `within`, also set `tolerance:` (relative tolerance).

Valid `ref:` prefixes: `planner.<field>`, `counters.<name>`, `fleet.<field>`, `overlay.<field>`.

---

## Adding a new γ scenario (< 3 hours)

1. Obtain or generate a mocker trace and place it in `traces/`:
   ```bash
   dynamo-mocker \
     --config examples/deployments/powerplanner/h200_disagg.yaml \
     --duration 120s \
     --dump-trace \
       components/src/dynamo/planner/tests/testbed/traces/my_trace.jsonl
   ```

2. Copy a γ base template:
   ```bash
   cp scenarios/_base/mocker_h200_disagg.yaml scenarios/G4_my_gamma_scenario.yaml
   ```

3. Set `mocker.trace_file: components/src/dynamo/planner/tests/testbed/traces/my_trace.jsonl`
   and add your `overlay.bias`, `events:`, and `assertions:`.

   The trace must be in **Mooncake format** (`timestamp` / `input_length` /
   `output_length` / `hash_ids` per line) — see `traces/README.md`.

4. Validate and run:
   ```bash
   pytest tests/testbed/tests/test_scenarios_loadable.py -v
   pytest "test_scenarios.py::test_gamma[G4_my_gamma_scenario]" -v
   ```

---

## Grafana dashboard

A pre-built Grafana dashboard JSON is in:
`grafana/testbed_dashboard.json`

It expects Prometheus textfile metrics written by:
```bash
python -m dynamo.planner.tests.testbed.runner --all \
    --prom-textfile /var/lib/node_exporter/textfile_collector/testbed.prom
```

Import into Grafana: **Dashboards → Import → Upload JSON file**.

Panels:
- `c_power_d` / `c_power_p` correction coefficients over ticks
- `cap_d` / `cap_p` applied power caps
- `n_d` / `n_p` replica counts
- `projected_w` vs `budget_w` (power budget utilization)
- `sweep_fired` events
- Per-scenario status heatmap

Filter by `scenario` label to view a single scenario or compare across them.

---

## Mutation gate (code review checklist)

If you rename a field in `recorder.py::TickSnapshot`, the `test_scenarios_loadable.py`
gate will fail at pytest collection and list every affected scenario — fix the
YAML references before merging.

If you change `aic_power_optimizer.py`'s EMA α coefficient, scenario A4
(`A4_step_drift_midstream.yaml`) should fail on its half-life assertion.
Run it explicitly before merging AIC tuning changes:
```bash
pytest "test_scenarios.py::test_alpha[A4_step_drift_midstream]" -v -s
```

---

## Hardware safety

The testbed has two layers of protection:

1. **Import guard** (`__init__.py`): installs a `sys.meta_path` finder that
   raises `ImportError` if anything tries to `import pynvml` _after_ the
   testbed package has been imported. Because the parent `dynamo.planner`
   package is usually imported first, this is best-effort and primarily
   documentary — production code never reaches `pynvml` anyway.

2. **K8s write block** (`conftest.py`): a session-scoped autouse fixture
   monkeypatches `kubernetes.client.CoreV1Api.patch_namespaced_pod` and
   `create_namespaced_pod` to raise `RuntimeError` for the entire pytest
   session. This is the actual safety net.

Neither layer affects production code; both are active only when running under
`pytest` inside this package.
