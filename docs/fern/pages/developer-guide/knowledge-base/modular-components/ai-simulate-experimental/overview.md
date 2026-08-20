---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AI Simulate (Experimental)
subtitle: Backend-neutral simulation and configuration-search tools
---

> [!WARNING]
> **Experimental.** AI Simulate is intended for evaluation and feedback, not production capacity
> planning. Its Python APIs, configuration schemas, search results, and deployment output may
> change without a standard deprecation period.

AI Simulate is a standalone Python distribution. It provides inference-engine forward-pass
simulation, deployment simulation, and search without depending on `ai-dynamo`.

For an engine-only single replay run, use `python -m aisimulate.replay`. For a replay with Dynamo
Router, Planner, or online adapters, use `python -m dynamo.replay`. Both commands share their base
replay configuration; Dynamo extends it with adapter options. The selected runtime validates each
`--*-engine-args` JSON payload, so runtime-specific fields can differ. For configuration search, call
`Sweeper(runner_factory=...).run(config)`.

## Migrate engine-only replay from Dynamo 1.5

Engine-only offline use of the Dynamo CLI and the `dynamo.replay.run_trace_replay` /
`run_synthetic_trace_replay` Python functions is deprecated in Dynamo 1.5.0 and is planned for
removal in Dynamo 1.6.0. The compatibility path still runs in 1.5.0. Dynamo Router, Planner, and
online replay entry points remain supported and do not emit this warning.

For the CLI, keep the shared base arguments and change the module. This example pins one
engine configuration that both runtimes accept, so the command is directly comparable:

```bash
ENGINE_ARGS='{"block_size":512,"num_gpu_blocks":100000,"aic_backend":"vllm","aic_system":"h200_sxm","aic_backend_version":"0.19.0","aic_tp_size":1,"aic_model_path":"Qwen/Qwen3-32B"}'

# Before: deprecated engine-only Dynamo path
python -m dynamo.replay \
  --input-tokens 1024 --output-tokens 128 --request-count 16 \
  --replay-concurrency 4 --extra-engine-args "$ENGINE_ARGS"

# Standalone replacement
python -m aisimulate.replay \
  --input-tokens 1024 --output-tokens 128 --request-count 16 \
  --replay-concurrency 4 --extra-engine-args "$ENGINE_ARGS"
```

For Python integrations, build an `aisimulate.ReplaySpec` and run it with the standalone factory:

```python
from aisimulate import (
    BackendDeploymentSpec,
    EngineReplayRunnerFactory,
    ReplaySpec,
)

engine_args = {
    "block_size": 512,
    "num_gpu_blocks": 100_000,
    "aic_backend": "vllm",
    "aic_system": "h200_sxm",
    "aic_backend_version": "0.19.0",
    "aic_tp_size": 1,
    "aic_model_path": "Qwen/Qwen3-32B",
}
spec = ReplaySpec(
    backend_deployment=BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.19.0",
        agg_engine_args=engine_args,
        num_workers=2,
    ),
    workload={"isl": 1024, "osl": 128, "request_count": 16},
    goal={"target": "throughput"},
    concurrency=4,
)

report = EngineReplayRunnerFactory().create(worker_id=0).run(spec)
print(report.metrics)
```

Keep using `DynamoReplayRunnerFactory` or `python -m dynamo.replay` when the replay selects Dynamo
Router or Planner adapters. Keep using the Dynamo command for `--replay-mode online`.

## Sweeper

[Sweeper](sweeper-experimental/overview.md) searches backend deployment settings against an injected replay runner.
Its core owns backend search, candidate orchestration, scoring, and the versioned `ReplaySpec`
contract.

Optional adapters extend the search without adding a Dynamo dependency to AI Simulate. The
`ai-dynamo` wheel registers the `dynamo.planner` and `dynamo.router` adapters. Selecting either
adapter imports its Dynamo implementation and adds a versioned runtime hook to the replay
specification.

KVBM search settings are deprecated and are not supported by the AI Simulate engine and replay
path. They have no adapter migration.

## Install

The `dynamo-planner` image installs the published `aisimulate==0.1.0.dev1` wheel from its local
wheelhouse. Dynamo builds `aisimulate-core==0.1.0-dev.1` from crates.io instead of vendoring the
AI Simulate source tree.

For Dynamo source development, install the published AI Simulate wheel, Dynamo, and the Planner
dependencies from the Dynamo repository root:

```bash
uv pip install "aisimulate==0.1.0.dev1"
uv pip install --no-deps -e .
uv pip install -r container/deps/requirements.planner.txt
```
