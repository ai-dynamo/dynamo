---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica
subtitle: Experimental backend-neutral configuration search
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search behavior, and output may change without a
> standard deprecation period. Spica does not guarantee service-level agreement (SLA) compliance,
> prediction accuracy, or globally optimal configurations.

Spica searches backend deployment configurations with a black-box optimizer. It materializes each
candidate as a versioned `ReplaySpec`, sends that specification to an injected replay
`RunnerFactory`, and returns ranked candidates or a Pareto front.

Spica does not import Dynamo. Dynamo-specific Planner and Router behavior lives in optional
adapters registered by the `ai-dynamo` wheel.

Use the `dynamo-planner` image when a sweep needs those adapters or the transitional Dynamo runner.
The image builds and installs AI Simulate and Dynamo from the same source revision; it does not
publish AI Simulate as a standalone wheel.

Spica uses AI Configurator's lower-layer forward-pass and memory providers, then evaluates each
candidate with the injected replay runner.

## Documentation

- [Overview](search-flow.md) describes ownership boundaries and the sweep flow.
- [Search Space](search-space.md) defines backend fields and adapter configuration.
- [Traffic](traffic.md) defines trace, request-rate, concurrency, and KV-load workloads.
- [Optimization Goals](optimization-goals.md) defines scalar and Pareto objectives.
- [Unrolled Samples](unrolled-samples.md) describes `BackendDeploymentSpec`, `ReplaySpec`, and candidate
  output.

## Configuration

The top-level schema contains five blocks:

```yaml
search_space:
  deployment_mode: [disagg, agg]
  backend: [vllm]
  model_name: deepseek-ai/DeepSeek-V3
  hardware_sku: h200_sxm
  gpu_budget: 32

adapters:
  dynamo.router:
    search_space:
      mode: [kv_router, round_robin]
  dynamo.planner:
    search_space:
      scaling_policy: [disabled, throughput_180_5, hybrid_180_5]

workload:
  trace_path: /data/replay/traffic.jsonl

goal:
  target: goodput_per_gpu
  sla: {ttft_ms: 2000, itl_ms: 30}

sweep:
  max_rounds: 40
  parallel_evals: 16
```

Each adapter receives its complete `search_space` mapping. It validates and expands that mapping
before the optimizer starts; it does not receive one concrete Planner or Router configuration.

## Python Entry Point

Pass a runner factory explicitly:

```python
from aisimulate.spica import SmartSearchConfig, run_smart_search
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = run_smart_search(
    config,
    runner_factory=DynamoReplayRunnerFactory(),
)
```

The standalone `python -m aisimulate.spica` command validates configuration but does not choose a
replay implementation.

## Optional Dynamo Features

The `ai-dynamo` wheel registers these package entry points:

| Adapter | Ownership | Runtime hook |
|---|---|---|
| `dynamo.planner` | Planner policies, load-predictor pre-sweep, and `PlannerConfig` materialization | `dynamo.planner:scaling_policy@1` |
| `dynamo.router` | Round-robin and KV-router search and materialization | `dynamo.router:placement_policy@1` |

The transitional `DynamoReplayRunnerFactory` translates those hooks to the current Dynamo Replay
API. Replay is undergoing the same dependency split in parallel: a backend-only Spica composition
will use the Dynamo-free replay runner, while a Dynamo composition will use the Dynamo replay
runner.

Planner performance queries use the `aiconfigurator-core` Python wheel directly and do not use a
`dynamo._core` engine-performance binding.

## Current Limitations

KV-capacity filtering and `kv_load_ratio` require an AI Configurator performance database for the
target `(hardware_sku, backend)`. Branch enumeration removes targets without one and backend or
deployment-mode combinations without a KV-feasible shape within the capacity budget. It raises
`NoViableParallelConfig` if no viable branch remains. Spica does not use the naive memory-estimation
fallback.

Treat workloads and search modes not covered by image smoke tests as unsupported experimental paths.
See [Traffic](traffic.md#kv_load_ratio-candidate-relative-concurrency) for the KV-load contract.

## Compatibility

- An adapter is imported only when its name appears under `adapters`.
- The runner advertises supported backend/topology pairs and runtime-hook versions before a study
  starts.
- TensorRT-LLM disaggregated replay remains excluded by the current Dynamo runner capability.
- KVBM fields are rejected with a migration error. Native G2 replaces KVBM; Spica does not forward
  old host/disk offload settings.
