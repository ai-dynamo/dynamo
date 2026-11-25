<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Duru calibration

This package measures isolated prefill and decode work for Duru. It has no
held-out or solo-session accuracy validation stage. It records raw observations,
uses the median of repeated measurements, and rejects malformed data or mismatched
runtime contracts. The measurement loop requires a caller-provided serving adapter.

## Point selection

Prefill cost is indexed by new tokens `P` and already cached tokens `D`. Supply the
resolved CUDA graph capture sizes from the engine that will serve requests. If its
mode supports prefill/mixed graphs, the planner samples `c-1`, `c`, and `c+1` around
each capture size `c` within the prefill budget, including the transition beyond
the largest captured size. Sparse geometric points cover the remaining range.
Prefix depths are sparse and aligned to the actual cache block alignment.

A capture size is not automatically a prefill boundary. Leave `--prefill-graphs`
off for eager or decode-only graph modes. Backend support and the resolved graph
mode determine which capture sizes are usable; the CLI does not discover these
from an unconfigured engine. Do not infer boundaries from a maximum graph size
alone or reuse another model's graph sizes.

Decode cost is indexed by context length `K`. A normal solo decode step schedules
one new token regardless of `K`, so graph capture sizes are not decode-context
boundaries. Decode contexts use a separate sparse grid.

For example, **only if these are your engine's resolved settings**:

```bash
PYTHONPATH=components/src python3 -m dynamo.duru_router.calibration plan \
  --cap 131072 --alignment 64 --max-prefill 8192 \
  --prefill-graphs --capture-sizes 1 2 4 8 16 32 64 128 256 512 \
  --out /path/to/run/plan.json
```

The command prints the point count before any measurement. `--prefix-points`,
`--decode-points`, `--repeats`, and `--warmups` control measurement cost. Defaults
are 10 measured repetitions and 3 warmups per point. Every request must actually
execute the planned prefill shape; backend chunk alignment can require adjusting
the plan. A shape mismatch is an error, not a measurement of a different point.

## Measurement adapter

`measure.measure` accepts these worker callbacks:

- `submit(prompt_ids, output_count, request_id, cache_salt)` returns `output_tokens`
  and `cached_tokens`, after the request finishes.
- `drain()` flushes the accepted-work JSONL ledger to disk.
- `reset()` clears prefix caches while the engine is idle.
- `ledger` points to the accepted-work journal. Each `accepted_output` record has
  `steps` with `client_request_id`, `accepted`, `generated_tokens`, `p`, `d`, `k`,
  `scheduled_at`, and `end`.

The caller owns the worker and must isolate its calibration traffic. Preserve
serving precision, tensor parallelism, attention backend, graph mode, chunk
budget, KV dtype/layout and offload configuration. The adapter must attach the
request ID to the scheduler observer so the ledger can reconcile accepted tokens.
Prefill measures schedule-to-accepted-output time; decode measures accepted-token
cadence after warmup, excluding first-token startup and the final event.

The CLI accepts an explicit adapter factory `module:factory`. The factory receives
the runtime contract and returns a context manager with `submit`, `drain`,
`reset`, `ledger`, and `vocab_size`. This local package includes the point planner
and measurement loop; a current-vLLM serving adapter is still an integration task.

```bash
PYTHONPATH=components/src python3 -m dynamo.duru_router.calibration measure \
  --plan /path/to/run/plan.json --contract /path/to/run/contract.json \
  --adapter your_serving_adapter:open_worker --out /path/to/run/calibration
```

The contract must include `timing_metric: "engine_service_time_v1"` and the actual
model/runtime/hardware settings. Each completed point is checkpointed. Resume
with `--resume`; the plan, vocabulary and contract must match. Raw samples are
retained. The output is `calibration.json`, without an accuracy gate.
Existing calibration artifacts remain loadable. To resume an older checkpoint,
reuse its saved plan: regenerating a plan after metadata cleanup changes the
checkpoint fingerprint even when the measurement cases are identical.

## Runtime use

In the live-score config, set `calibration` to the resulting JSON and
`runtime_contract` to the serving worker's actual contract object. The loader
checks equality with the calibration contract and the table shape. Generate this
contract from the worker; do not copy a calibration contract to disguise a mismatch.
The live predictor uses the calibrated prefill budget for five-step lookahead.

Both calibration consumers and live scoring use `predict.py`. It interpolates
within graph regimes and never blends across a declared jump. Outside the measured
outer range, live scoring uses edge-slope extrapolation;
nonpositive or nonfinite predictions remain unavailable scores. This does not
establish prediction accuracy outside the measured range.
