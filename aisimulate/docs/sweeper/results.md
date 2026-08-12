---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Results
subtitle: Replay specifications, ranked candidates, and Pareto fronts
---

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
