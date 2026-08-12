---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Results
subtitle: Replay specifications, ranked candidates, and Pareto fronts
---

<!--
Generated from `aisimulate/docs/sweeper/results.md` by `docs/fern/scripts/sync_aisimulate_docs.py`.
Edit the canonical source instead of this Fern copy.
-->

> [!WARNING]
> **Experimental.** Sweeper's replay and result contracts may change without a standard deprecation
> period.

Sweeper materializes every optimizer suggestion in the main process. It unrolls the backend
selection, asks each configured provider for its concrete adapter configuration and runtime hooks,
then constructs a `ReplaySpec`.

## Replay Specification

`ReplaySpec` version 1 contains:

- a `BackendDeploymentSpec` with topology, backend version, engine arguments, and worker counts;
- the validated workload and optimization goal;
- concrete concurrency when KV-load search derives it;
- concrete adapter configurations and their runtime hooks.

`RunnerCapabilities.require_compatible` checks the version, backend/topology pair, and hooks before
execution. `canonical_json` creates deterministic strict JSON and rejects non-finite values.

## Candidate Output

For a scalar goal, `Sweeper.run` returns feasible `Candidate` objects sorted best-first. Each
candidate contains:

| Field | Meaning |
|---|---|
| `config` | unrolled backend sample plus nested concrete adapter configuration |
| `used_gpus` | total GPUs assigned to the deployment |
| `metrics` | normalized values returned by replay |
| `score` | objective normalized so larger is better |
| `objectives` | raw per-objective values for Pareto searches; otherwise `None` |

For `goal.target: pareto`, the result contains only non-dominated candidates and preserves each
objective's natural direction.

```python
candidates = sweeper.run(config)
best = candidates[0]
print(best.config)
print(best.metrics)
```

Exact repeated suggestions reuse a result from the current `run` call. The cache does not persist
between calls, even when the same `Sweeper` instance is reused.

## CLI Output

The CLI exposes the same result contract in human- and machine-readable forms:

```bash
# Complete JSON envelope on standard output
aisimulate recommend --stack engine --config sweep.yaml --output json

# Persist JSON plus a flattened CSV summary
aisimulate recommend --stack engine --config sweep.yaml --output-dir results --top-n 5
```

`sweep_results.json` carries a schema version, result type, goal, and every returned `Candidate`
without flattening. A scalar search also writes `best_config_topn.csv`. A Pareto search writes the
complete non-dominated set to `pareto.csv`; it does not manufacture a top-N ordering for a front
whose points are mutually non-dominated.

## Legacy Workflow Differences

The new Sweeper preserves the legacy AIConfigurator behaviors that belong to configuration search:
engine and optional Dynamo execution, backend/topology comparison, ranked or Pareto results, top-N
scalar recommendations, strict aggregate SLA filtering, concise validation failures, and saved
machine-readable output.

The following workflows intentionally remain outside Sweeper:

- support-matrix queries and single-point static estimates;
- deployment artifact generation;
- one CLI flag per modeling knob and exhaustive experiment-file execution.

Those workflows have different ownership and should consume the versioned result contract rather
than add legacy-only translation paths to Sweeper.
