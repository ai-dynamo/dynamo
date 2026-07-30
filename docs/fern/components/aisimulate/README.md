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

## Spica

[Spica](spica/README.md) searches backend deployment settings against an injected replay runner.
Its core owns backend search, candidate orchestration, scoring, and the versioned `ReplaySpec`
contract.

Optional adapters extend the search without adding a Dynamo dependency to AI Simulate. The
`ai-dynamo[simulation]` extra installs Planner simulation dependencies and publishes
`dynamo.planner` and `dynamo.router` adapters. Selecting either adapter imports its Dynamo
implementation and adds a versioned runtime hook to the replay specification.

KVBM search settings are deprecated and have no adapter migration. Native G2 replaces that path.

## Install

Install the AI Simulate distribution from the repository root:

```bash
uv pip install -e ./aisimulate
```

To use the Dynamo adapters and the transitional Dynamo replay runner, also install the matching
`ai-dynamo` wheel.
