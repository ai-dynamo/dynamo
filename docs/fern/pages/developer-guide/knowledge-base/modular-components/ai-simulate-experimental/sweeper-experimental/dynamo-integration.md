---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Dynamo Sweeper Integration
subtitle: Install and compose Planner, Router, and Dynamo Replay with AISimulate
---

> [!WARNING]
> **Experimental.** The AISimulate CLI, adapter ABI, and replay contracts may change without a
> standard deprecation period.

AISimulate's Sweeper core does not depend on Dynamo. Dynamo owns its optional Planner and Router
sweep configuration providers and `DynamoReplayRunnerFactory`.

## Install

AISimulate requires Python 3.11 through 3.13. The `ai-dynamo` package remains installable on Python
3.10, but the AISimulate dependency and CLI are not installed there.

The supported prebuilt environment is the `dynamo-planner` image. The image stages and installs the
published `aisimulate==0.1.0.dev2` wheel alongside the Dynamo wheels. Dynamo's Rust workspace
resolves `aisimulate-core==0.1.0-dev.2` from crates.io.

For Dynamo source development, install the published AISimulate wheel and build the matching
Dynamo bindings:

```bash
python3 -m pip install pip "maturin[patchelf]"
cd lib/bindings/python
maturin develop --uv --release --features aic-forward-pass
cd ../../..
python3 -m pip install --no-deps -e .
python3 -m pip install "aisimulate==0.1.0.dev2"
python3 -m pip install -r container/deps/requirements.planner.txt
```

On supported Python versions, `ai-dynamo` declares the exact AISimulate release as a base
dependency. No `ai-dynamo[simulation]` extra is required.

## Run the Unified CLI

Select the Dynamo stack explicitly to load the runner plus any configured Router and Planner
adapters:

```bash
aisimulate predict --stack dynamo --config prediction.yaml
aisimulate recommend --stack dynamo --config recommendation.yaml
```

The `--stack` option defaults to `engine`. Without `--stack dynamo`, top-level `router` and
`planner` sections have no matching adapter and fail configuration resolution.

The `ai-dynamo` distribution registers:

| Entry-point group | Name | Purpose |
|---|---|---|
| `aisimulate.runner_factories` | `dynamo` | Execute a complete replay through Dynamo composition |
| `aisimulate.config_adapters` | `dynamo.router` | Validate and materialize the public `router` section |
| `aisimulate.config_adapters` | `dynamo.planner` | Validate and materialize the public `planner` section |
| `aisimulate.sweep_config_providers` | `dynamo.router`, `dynamo.planner` | Preserve the legacy Sweeper Python API |

## Use the Python SDK

Call the retained Sweeper Python API only when an application needs the legacy `SmartSearchConfig`
contract and explicit runner injection:

```python
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory

config = SmartSearchConfig.from_yaml("smart_sweep.yaml")
candidates = Sweeper(
    runner_factory=DynamoReplayRunnerFactory(),
).run(config)
```

There is no compatibility CLI alias or `--enable-dynamo` flag.

## Dynamo Providers

The `ai-dynamo` distribution registers these entry points:

| Adapter | Provider | Runtime hook |
|---|---|---|
| `dynamo.planner` | `DynamoPlannerSweepConfigProvider` | `dynamo.planner:scaling_policy@1` |
| `dynamo.router` | `DynamoRouterSweepConfigProvider` | `dynamo.router:placement_policy@1` |

The public recommendation YAML places the adapter-owned domains at the top level:

```yaml
router:
  policy: {choices: [round_robin, kv_router]}
  prefill_load_model: {type: none}
  overlap_score_credit: {choices: [0.0, 0.5, 1.0]}
planner:
  policy: enabled
  scaling_policy: {preset: [disabled, load_180_5]}
  fpm_sampling: {preset: [default]}
  load_sensitivity: {preset: [default]}
  load_predictor: {preset: [arima_raw]}
```

The legacy `SmartSearchConfig.adapters` mapping remains available only through the Python SDK.

> [!WARNING]
> The legacy flat Planner preset lists remain accepted for backward compatibility but emit a
> `FutureWarning`. They will be removed after the 1.5 release. Nest each list under its sub-item's
> `preset` field.

Each Planner preset sub-item owns a complete knob set:

| Sub-item | Knobs covered by every preset |
|---|---|
| `scaling_policy` | Throughput/load enablement and both adjustment intervals |
| `fpm_sampling` | Maximum FPM samples and sample-bucket size |
| `load_sensitivity` | Scale-down sensitivity and minimum observations |
| `load_predictor` | Predictor family, log transform, Prophet window, and all Kalman parameters |

Named presets and custom mappings are validated against the complete sub-item. The provider fills
family defaults for conditionally inactive predictor knobs before validation.

The Planner provider derives load-predictor parameters from all configured scaling intervals during
`generate_search_space`. Enabled candidates materialize a concrete `PlannerConfig` runtime hook.
The Router provider materializes either round-robin behavior without a hook or a concrete KV-router
hook.

## Replay Composition

`DynamoReplayRunnerFactory` converts each serializable `ReplaySpec` into an invocation of the
shared AISimulate Replayer. It resolves materialized Planner and Router hooks into Dynamo-owned
scaling and placement policies, while the Replayer continues to own traffic execution and report
generation.

| Load | No Planner hook | Dynamo Planner hook |
|---|---|---|
| Mooncake trace | Replayer with no scaling | Replayer with the selected Planner scaling policy |
| Synthetic traffic | Replayer with no scaling | Replayer with the selected Planner scaling policy |

The runner passes trace, fixed, or KV-load-derived closed-loop concurrency through
`ReplaySpec.concurrency`. The public compiler maps `evaluation.sla` into `ReplaySpec.goal.sla`, and
`DynamoReplayRunnerFactory` reads that lowered field as the replay goodput SLA. This SLA is
independent of the Planner's scaling SLA. A `dynamo.router:placement_policy@1` hook selects the
Dynamo placement policy. Without that hook, the Replayer uses its built-in round-robin policy.
