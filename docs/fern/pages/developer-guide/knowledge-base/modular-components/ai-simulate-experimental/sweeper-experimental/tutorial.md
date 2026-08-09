---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper Tutorial
subtitle: Configure, execute, and inspect a replay-backed search
---

<!--
Generated from `aisimulate/docs/sweeper/tutorial.md` by `docs/fern/scripts/sync_aisimulate_docs.py`.
Edit the canonical source instead of this Fern copy.
-->

> [!WARNING]
> **Experimental.** Sweeper's API and search behavior may change without a standard deprecation
> period.

## 1. Define the Backend Search

Choose a model, hardware system, deployment modes, backends, and GPU budget:

```yaml
search_space:
  model_name: meta-llama/Meta-Llama-3.1-8B
  hardware_sku: gb200
  deployment_mode: [agg]
  backend: [trtllm]
  gpu_budget: 8
```

Sweeper enumerates legal parallel configurations, removes unsupported runner topologies, and adds
the active engine-role knobs to the optimizer study.

## 2. Define One Workload and Goal

```yaml
workload:
  isl: 1024
  osl: 128
  request_rate: 4
  num_request_ratio: 10

goal:
  target: throughput
```

Every candidate is evaluated against this workload. See [Traffic](traffic.md) for trace and
closed-loop alternatives, and [Optimization Goals](optimization-goals.md) for SLA and Pareto modes.

## 3. Control the Sweep

```yaml
sweep:
  max_rounds: 2
  candidates_per_round: 4
  parallel_evals: 2
  max_eval_seconds: 300
```

Each round is a barrier: the sampler asks for unique candidates, replay evaluates them, and then
the scores are reported back together. `parallel_evals` controls replay worker fan-out.

## 4. Add Optional Feature Search

An installed or injected provider owns the schema below its adapter name:

```yaml
adapters:
  example.policy:
    search_space:
      mode: [balanced, latency]
```

The provider receives the complete `search_space` mapping. It does not receive a preselected
concrete feature configuration.

## 5. Run and Inspect

```python
sweeper = Sweeper(
    runner_factory=my_runner_factory,
    providers={"example.policy": my_provider},
)
candidates = sweeper.run(config)

for candidate in candidates[:5]:
    print(candidate.score, candidate.used_gpus, candidate.config)
```

The same `Sweeper` instance can run multiple configurations. Studies, caches, and process pools are
new for every call.
