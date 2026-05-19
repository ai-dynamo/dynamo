# Testbed Traces

This directory holds `dynamo-mocker` traces for the γ-class (replay-based)
scenarios (`G1`, `G2`, `G3`).

## Format — Mooncake JSONL (what `PlannerReplayBridge` actually consumes)

Traces are `.jsonl` files in the **Mooncake request-trace format** that the
mocker scheduler consumes via `PlannerReplayBridge.{create_disagg,
from_trace_file_disagg}`. Every line is one *request* to be issued at the
specified simulated time:

```json
{"timestamp": 0,     "input_length": 512, "output_length": 50, "hash_ids": [0]}
{"timestamp": 12000, "input_length": 512, "output_length": 50, "hash_ids": [1]}
{"timestamp": 24000, "input_length": 512, "output_length": 50, "hash_ids": [2]}
```

Required fields per line:

| Field | Type | Meaning |
|-------|------|---------|
| `timestamp` | int | Wall-clock milliseconds at which the request arrives (relative to trace start) |
| `input_length` | int | Prompt length in tokens |
| `output_length` | int | Decode length in tokens |
| `hash_ids` | list[int] | KV-block hash IDs for prefix-cache routing. One per `trace_block_size` block of input. |

The bridge's `arrival_speedup_ratio` scales `timestamp` (e.g., 2.0 = arrivals
happen at half the wall-clock spacing). The engine's own `speedup_ratio` (set
in `MockEngineArgs`) scales the simulated processing time independently —
see Appendix D.5 of `powerplanner-testbed-design.md` for why this matters
for short test runs.

> **Historical note**: an older internal harness used a different per-line
> format (FPM snapshots: `ts_ns`, `prefill_workers`, `decode_workers`). That
> format is **not** consumed by `PlannerReplayBridge` and produces
> `trace line N is missing hash_ids` if pointed at the current bridge. The
> placeholder bundled here has been regenerated in Mooncake format —
> see Appendix D.4.

## Provided Traces

| File | Description | Requests | Duration |
|------|-------------|---------:|---------:|
| `placeholder_h200_disagg_1rps.jsonl` | Synthetic placeholder for the older `create_disagg` bridge fallback. 300 small requests (`isl=512`, `osl=50`), one unique hash per request, staggered every 12 s. Used when a scenario specifies `synthetic_workload: true` but the installed bridge has no `from_synthetic_disagg` constructor. | 300 | 3588 s |

> **Note**: Real captured traces are not committed to this repository due to
> size. Generate them with `dynamo-mocker` against the `h200-disagg`
> deployment config and place them here. The `ScenarioSpec.mocker.trace_file`
> field accepts an absolute path or a path relative to repo root.

## Why the placeholder is small + sparse

The placeholder is sized to keep the γ-class suite under 10 seconds of
wall-time, not to drive realistic load. With 300 small requests over
3600 simulated seconds at 1-worker capacity, fleet utilization is ≈ 0.003 —
enough to exercise the bridge's tick/scaling loop and the planner's
`reconcile_fpm_worker_count` path, but **not** enough to drive AIC EMA
correction. That's why `test_alpha_gamma_agree_on_decode_drift` auto-skips
when only the older bridge API is available (Appendix D.7).

When the newer `from_synthetic_disagg` API is built into the dev image,
γ-class will drive proportional load through the synthetic-workload generator
and this placeholder will only be used in true `from_trace_file_disagg`
scenarios.

## Generating Traces

```bash
dynamo-mocker \
  --config examples/deployments/powerplanner/h200_disagg.yaml \
  --duration 120s \
  --dump-trace traces/my_trace.jsonl
```

Then reference the trace in your scenario:

```yaml
mocker:
  trace_file: components/src/dynamo/planner/tests/testbed/traces/my_trace.jsonl
  synthetic_workload: false
```
