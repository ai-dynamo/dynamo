---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Dynamo Sweeper Integration
subtitle: Install and compose Planner, Router, and Dynamo Replay with AI Simulate
---

> [!WARNING]
> **Experimental.** The AI Simulate and replay dependency split is still in progress.

AI Simulate's Sweeper core does not depend on Dynamo. Dynamo owns its optional Planner and Router
sweep configuration providers and `DynamoReplayRunnerFactory`.

## Install

The supported prebuilt environment is the `dynamo-planner` image. The image builds AI Simulate and
Dynamo from the same source revision and installs both wheels. The AI Simulate wheel remains inside
the image and is not published as a release artifact.

For source development, build the matching Dynamo bindings and install both source distributions:

```bash
python -m pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass
cd ../../..
python -m pip install --no-deps -e .
python -m pip install -e ./aisimulate
python -m pip install -r container/deps/requirements.planner.txt
```

The repository does not yet provide an `ai-dynamo[simulation]` extra. Add that extra only after the
full AI Simulate and Replay refactor has a supported distribution path.

## Run a Dynamo Sweep

```bash
python aisimulate/examples/sweeper/tools/run_sweep.py \
  --config aisimulate/examples/sweeper/configs/smart_sweep.yaml
```

Or compose the runtime directly:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=DynamoReplayRunnerFactory(),
).run(config)
```

There is no `--enable-dynamo` flag. Selecting a Dynamo adapter in configuration and injecting the
Dynamo runner are the explicit composition points.

## Dynamo Providers

The `ai-dynamo` distribution registers these entry points:

| Adapter | Provider | Runtime hook |
|---|---|---|
| `dynamo.planner` | `DynamoPlannerSweepConfigProvider` | `dynamo.planner:scaling_policy@1` |
| `dynamo.router` | `DynamoRouterSweepConfigProvider` | `dynamo.router:placement_policy@1` |

Each adapter value is a search space, not one concrete Planner or Router config:

```yaml
adapters:
  dynamo.router:
    search_space:
      mode: [kv_router, round_robin]
      overlap_score_credit: [0.0, 0.5, 1.0]
  dynamo.planner:
    search_space:
      scaling_policy: [disabled, load_180_5]
      fpm_sampling: [default]
      load_sensitivity: [default]
```

The Planner provider derives load-predictor parameters from all configured scaling intervals during
`generate_search_space`. Enabled candidates materialize a concrete `PlannerConfig` runtime hook.
The Router provider materializes either round-robin behavior without a hook or a concrete KV-router
hook.

## Replay Composition

`DynamoReplayRunnerFactory` converts each serializable `ReplaySpec` into an invocation of the
shared AI Simulate Replayer. It resolves materialized Planner and Router hooks into Dynamo-owned
scaling and placement policies, while the Replayer continues to own traffic execution and report
generation.

| Load | No Planner hook | Dynamo Planner hook |
|---|---|---|
| Mooncake trace | Replayer with no scaling | Replayer with the selected Planner scaling policy |
| Synthetic traffic | Replayer with no scaling | Replayer with the selected Planner scaling policy |

The runner passes trace, fixed, or KV-load-derived closed-loop concurrency through
`ReplaySpec.concurrency`. It applies a goodput SLA only when `goal.sla` is configured; this SLA is
independent of the Planner's scaling SLA. A `dynamo.router:placement_policy@1` hook selects the
Dynamo placement policy. Without that hook, the Replayer uses its built-in round-robin policy.
