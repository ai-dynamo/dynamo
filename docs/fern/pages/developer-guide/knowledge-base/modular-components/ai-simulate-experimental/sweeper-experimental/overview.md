---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweeper
subtitle: Experimental backend-neutral configuration search
---

> [!WARNING]
> **Experimental.** Sweeper is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search behavior, and output may change without a
> standard deprecation period.

> [!NOTE]
> **Retained Python SDK.** This section documents `SmartSearchConfig` and `Sweeper.run` for
> applications that inject a replay runtime. For new command-line workflows, start with
> [Sweep DynoSim Configurations](../../../../../cli/operations/simulation-with-dynosim/dynosim-sweeps.mdx)
> and `aisimulate recommend --stack dynamo`. See [SDK and CLI
> configuration](#sdk-and-cli-configuration) before reusing an SDK YAML file with the unified CLI.

Sweeper searches deployment configurations with a black-box optimizer. It turns every suggestion
into a versioned `ReplaySpec`, sends that specification to an injected `RunnerFactory`, and returns
ranked candidates or a Pareto front.

The `aisimulate` package owns only backend-neutral simulation behavior. Optional feature packages
can register a `SweepConfigProvider` that contributes search dimensions and materializes its part
of a replay. Sweeper imports a provider only when its adapter name appears in the configuration.

## Start Here

- [Quickstart](quickstart.md) runs a small backend-neutral sweep.
- [Tutorial](tutorial.md) explains a complete sweep configuration.
- [Architecture](architecture.md) shows the provider, replay, and worker boundaries.
- [Configuration](configuration.md) describes core and adapter-owned search spaces.
- [Traffic](traffic.md) defines trace, request-rate, concurrency, and KV-load workloads.
- [Optimization Goals](optimization-goals.md) defines scalar and Pareto objectives.
- [Results](results.md) describes `ReplaySpec` and `Candidate` output.
- [Sweep Configuration Providers](sweep-config-provider.md) documents the extension ABI.

## Python Entry Point

`Sweeper.run` executes the retained Python SDK configuration. Supply a replay runtime explicitly:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper

config = SmartSearchConfig.from_yaml("sweep.yaml")
candidates = Sweeper(runner_factory=my_runner_factory).run(config)
```

For the public YAML contract and stack discovery, use `aisimulate recommend --config
recommendation.yaml`. The `Sweeper` class remains available for callers that need the legacy Python
SDK configuration and an explicitly injected runtime.

## SDK and CLI Configuration

| Interface | Configuration and execution |
|---|---|
| Retained Python SDK | `SmartSearchConfig` with `search_space`, `workload`, `goal`, `sweep`, and optional `adapters`; the application supplies a `RunnerFactory` |
| Unified CLI | Public recommendation YAML with `engine`, `optimization`, and optional `traffic`, `evaluation`, and `optimizer`; `--stack dynamo` loads Dynamo's runner and top-level `router` and `planner` adapters |

The SDK YAML examples in this section are not input files for `aisimulate recommend --config`.
Re-express the workload, search domains, and objectives using the [recommendation
reference](../../../../../reference/components/dynosim-sweep-reference.mdx), then validate the
result with the unified CLI. The SDK remains available for existing integrations and explicit
runner injection; this guidance does not deprecate Dynamo's Router or Planner providers.

## Compatibility

- A provider is imported only when its adapter name appears under `adapters`.
- The runner advertises supported `ReplaySpec` versions, backend/topology pairs, and runtime hooks
  before a study starts.
- Every `Sweeper.run` call owns fresh optimizer studies, result caches, runners, and worker pools.
- KVBM search fields are rejected. The AISimulate engine and replay path do not support those
  fields and provide no adapter migration for the old host or disk offload settings.
