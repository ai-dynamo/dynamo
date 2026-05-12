# G3 Behavior Parity Fixtures

Golden output fixtures for the **G3 行为等价矩阵** (Behavior Parity Test
Matrix).

## Purpose

Lock the current `PlannerStateMachine.on_tick` output as **永久 golden
source**. After PSM is decomposed into builtin plugins (PR 6), the
plugin-driven path must produce **位级一致** output for the same input
sequences—otherwise behavior parity (G3) is broken.

## Fixture Layout

```
g3_fixtures/
├── __init__.py
├── README.md             # this file
├── serializers.py        # JSON encode/decode for TickInput/PlannerEffects/...
├── scenarios.py          # representative (config, ticks) cases
├── dump_tool.py          # CLI: generate / verify
└── golden/               # generated JSONL files (committed to repo)
    ├── baseline_disagg_throughput_only_sla.jsonl
    ├── disagg_load_throughput_sla.jsonl
    ├── disagg_throughput_only_latency_easy.jsonl
    ├── agg_throughput_only_sla.jsonl
    ├── prefill_throughput_only_sla.jsonl
    └── decode_throughput_only_sla.jsonl
```

## Usage

### Generate fixtures (run from repo root)

```bash
cd components/src/dynamo
python -m planner.tests.plugins.g3_fixtures.dump_tool
```

This writes one JSONL per scenario into `golden/`.

### List scenarios

```bash
python -m planner.tests.plugins.g3_fixtures.dump_tool --list
```

### Single scenario

```bash
python -m planner.tests.plugins.g3_fixtures.dump_tool \
    --scenario baseline_disagg_throughput_only_sla
```

### Verify mode (re-run + compare; no write)

Used by CI to detect any drift between current PSM behavior and locked fixtures:

```bash
python -m planner.tests.plugins.g3_fixtures.dump_tool --verify
```

Exit code:
- `0`: all scenarios match.
- `1`: at least one scenario diverges (CI fail).
- `2`: scenario name unknown.

## Fixture File Format

JSONL (one JSON object per line). First line is metadata header,
subsequent lines are tick records.

### Header (line 1)

```json
{"_meta": {
    "fixture_format_version": 1,
    "git_commit": "<sha at fixture lock time>",
    "python_version": "3.13.x",
    "scenario": "baseline_disagg_throughput_only_sla",
    "description": "...",
    "config": {...},
    "caps": {...},
    "initial_tick_at_s": 0.0,
    "initial_tick": {...},
    "num_ticks": 3
}}
```

### Tick record (lines 2..N+1)

```json
{
    "tick_index": 0,
    "now_s": 5.0,
    "tick_input": {
        "now_s": 5.0,
        "traffic": null,
        "worker_counts": {"ready_num_prefill": 1, "ready_num_decode": 1, ...},
        "fpm_observations": {
            "prefill": {"w1:0": "<base64 msgspec FPM>"},
            "decode":  {"w1:0": "<base64 msgspec FPM>"}
        }
    },
    "scheduled_tick": {...},
    "planner_effects": {
        "scale_to": {"num_prefill": 2, "num_decode": 3} | null,
        "next_tick": {...} | null,
        "diagnostics": {
            "estimated_ttft_ms": 480.0 | null,
            ...
            "load_decision_reason": "scale_up" | null,
            ...
        }
    }
}
```

### FPM Encoding

`ForwardPassMetrics` payloads use **msgspec JSON + base64**—preserves
the wire format used in production. Decoder in
`serializers.decode_fpm`.

## Fixture Lock Protocol

**关键**: fixtures must be locked at a stable git commit before PSM is
modified (PR 5+). The lock procedure:

1. **Verify current main is clean and PSM is unmodified**:
   ```bash
   git status
   git log --oneline -5 components/src/dynamo/planner/core/state_machine.py
   ```
2. **Generate fixtures**:
   ```bash
   cd components/src/dynamo
   python -m planner.tests.plugins.g3_fixtures.dump_tool
   ```
3. **Commit golden files** to repo (this PR):
   ```bash
   git add components/src/dynamo/planner/tests/plugins/g3_fixtures/golden/
   git add components/src/dynamo/planner/tests/plugins/g3_fixtures/*.py
   git add components/src/dynamo/planner/tests/plugins/g3_fixtures/README.md
   git commit -m "Lock G3 behavior parity fixtures (Pre-PR 5)"
   ```
4. **Tag the commit**:
   ```bash
   git tag pre-plugin-architecture
   # Push tag to remote (manual; no force-push to remote tags allowed):
   git push origin pre-plugin-architecture
   ```

After this point:
- **PR 5/6/7/8 sub-tasks must be read-only on PSM** — any change to
  `core/state_machine.py` / `core/load_scaling.py` / `core/throughput_scaling.py`
  may invalidate the fixtures and require re-lock.
- The dump tool's `--verify` mode is run on every CI build to catch
  unintentional PSM drift.

## Adding a New Scenario

1. Edit `scenarios.py`:
   ```python
   def _my_new_scenario() -> Scenario:
       return Scenario(
           name="my_new_scenario",
           description="...",
           config_overrides=dict(...),
           bootstrap_fn=lambda core: ...,
           ticks=[
               TickInput(now_s=..., ...),
               ...
           ],
       )

   ALL_SCENARIOS = [..., _my_new_scenario()]
   ```
2. Re-run dump tool to generate the fixture file:
   ```bash
   python -m planner.tests.plugins.g3_fixtures.dump_tool \
       --scenario my_new_scenario
   ```
3. Commit the new fixture file along with the scenario code change.

## G3 Matrix Coverage Status

The full G3 matrix is `mode × (enable_load, enable_throughput) ×
optimization_target = 4 × 3 × 3 = 36` cells. The v1 fixture set
covers **representative cases** rather than every cell exhaustively.
Cells that are redundant or invalid are skipped (e.g. `prefill` mode
with `enable_load=False, enable_throughput=False` is a no-op).

| Scenario | mode | enable_load | enable_throughput | opt_target |
|---|---|---|---|---|
| `baseline_disagg_throughput_only_sla` | disagg | ✗ | ✓ | sla |
| `disagg_load_throughput_sla` | disagg | ✓ | ✓ | sla |
| `disagg_load_only_latency_easy` | disagg | ✓ | ✗ | latency |
| `agg_throughput_only_sla` | agg | ✗ | ✓ | sla |
| `prefill_throughput_only_sla` | prefill | ✗ | ✓ | sla |
| `decode_throughput_only_sla` | decode | ✗ | ✓ | sla |

**Note**: `easy mode` (optimization_target ≠ "sla") + `enable_throughput_scaling=True`
is an **invalid combination** —— PSM doesn't instantiate predictor in
easy mode and crashes when traffic is observed. Easy mode is load-only
by design.

**v1 covers 6 of ~30 valid cells**. PR 6 expands the matrix to full coverage
during 6-8 sub-task (G3 行为等价矩阵完整 sweep). Adding scenarios is
cheap—each new scenario is a single function in `scenarios.py`.

## CI Integration

### Pre-PR 5 fixture lock guard (active now)

`test_g3_fixture_parity.py` (in this same directory) re-runs each
scenario through the **current `PlannerStateMachine`** and asserts the
output matches the locked fixture byte-for-byte. It carries the markers

```python
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]
```

so it's automatically picked up by the existing **`planner-test`** job
in `.github/workflows/pr.yaml` (filter: `pre_merge and planner and gpu_0`,
container: `target: planner`). **No CI workflow changes are needed.**

What this guards against:
- PR 5 / 7 sub-tasks accidentally modifying PSM code paths that are
  supposed to be read-only until PR 11 cleanup.
- Any unrelated PR touching `core/state_machine.py` /
  `core/load_scaling.py` / `core/throughput_scaling.py` causing
  unintentional behavior drift.

Failure mode: clear, parametrized per scenario. The assertion message
points the developer at the README protocol for re-locking if the
change was intentional.

### PR 6 — orchestrator parity test (planned)

PR 6 6-8 sub-task adds a sibling test that swaps the SUT from PSM to
the new orchestrator + real builtin plugins:

```python
# tests/plugins/builtins/test_g3_real_parity.py
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_g3_parity_real_builtins(scenario):
    """Run scenario through orchestrator + real builtin plugins;
    assert byte-level match with locked fixture."""
    ...
```

Same markers, same `planner-test` job — both tests will run side-by-side,
giving us the strongest possible parity guarantee (PSM → fixture →
orchestrator).

### PR 11 — fixture re-lock (planned)

PR 11 (cleanup) deletes PSM. At that point `test_g3_fixture_parity.py`
is rewritten to drive the orchestrator instead, and the
`pre-plugin-architecture` git tag becomes a historical reference.
