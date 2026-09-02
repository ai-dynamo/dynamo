---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AISimulate (Experimental)
subtitle: Backend-neutral simulation and configuration-search tools
---

> [!WARNING]
> **Experimental.** AISimulate is intended for evaluation and feedback, not production capacity
> planning. Its Python APIs, configuration schemas, search results, and deployment output may
> change without a standard deprecation period.

AISimulate is a standalone Python distribution. It provides inference-engine forward-pass
simulation, deployment simulation, and search without depending on `ai-dynamo`.

Use `aisimulate predict --stack engine` for an engine-only prediction. Use `aisimulate predict
--stack dynamo` to add Dynamo Router and Planner adapters. Both commands read the same YAML schema;
the `ai-dynamo` package owns and validates the optional `router` and `planner` sections. Use
`aisimulate recommend` with the same stack selection to search configuration domains.

> [!WARNING]
> The former Dynamo online replay adapter has no replacement in the unified CLI yet. `aisimulate
> predict` and `aisimulate recommend` are offline-only. Online replay will return in a future
> release.

## Sweeper

[Sweeper](sweeper-experimental/overview.md) searches backend deployment settings against an injected replay runner.
Its core owns backend search, candidate orchestration, scoring, and the versioned `ReplaySpec`
contract.

Optional adapters extend the search without adding a Dynamo dependency to AISimulate. The
`ai-dynamo` wheel registers the `dynamo.planner` and `dynamo.router` adapters. Selecting either
adapter imports its Dynamo implementation and adds a versioned runtime hook to the replay
specification.

KVBM search settings are deprecated and are not supported by the AISimulate engine and replay
path. They have no adapter migration.

## Install

AISimulate requires Python 3.11 through 3.13. Dynamo itself still supports Python 3.10, but the
`aisimulate` dependency and its CLI are not installed in a Python 3.10 environment.

The `dynamo-planner` image installs the published `aisimulate==0.1.0.dev2` wheel from its local
wheelhouse. Dynamo builds `aisimulate-core==0.1.0-dev.2` from crates.io instead of vendoring the
AISimulate source tree.

For Dynamo source development, install the published AISimulate wheel, Dynamo, and the Planner
dependencies from the Dynamo repository root:

```bash
python3 -m pip install "aisimulate==0.1.0.dev2"
python3 -m pip install --no-deps -e .
python3 -m pip install -r container/deps/requirements.planner.txt
```
