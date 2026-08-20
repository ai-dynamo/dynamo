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

For the CLI, keep the shared base arguments and change the module:

```bash
# Before: deprecated engine-only Dynamo path
python -m dynamo.replay \
  --input-tokens 1024 --output-tokens 128 --request-count 16 \
  --replay-concurrency 4 --extra-engine-args '{"block_size":64}'

# Standalone replacement
python -m aisimulate.replay \
  --input-tokens 1024 --output-tokens 128 --request-count 16 \
  --replay-concurrency 4 --extra-engine-args '{"block_size":64}'
```

For Python integrations, build an `aisimulate.ReplaySpec` and run it with the standalone factory:

```python
from aisimulate import EngineReplayRunnerFactory

report = EngineReplayRunnerFactory().create(worker_id=0).run(spec)
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
